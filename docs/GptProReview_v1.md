A) Top Missing Methodologies (max 12)

| Methodology                                                                                                 | Why it matters for our exact setup                                                                                                                                                                                                                       | Evidence from which paper/doc (with page/section)                                                                                                                                                                                                                                                                                                                              | Expected impact (accuracy/latency/robustness)               | Implementation cost (Low/Med/High) | Priority (P0/P1/P2) | Where to update in PRD/spec (exact section names)                                                                                                                |
| ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- | ---------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1) **부분 라벨(Partial labels) 처리: task별 `label_available` 플래그 + ignore-index 마스크 + loss masking**              | 현재 변환 스펙은 “라벨이 없는 태스크”를 **0(배경)** 으로 채워 넣는 규칙이 존재합니다. 이대로 학습하면 RLMD/Cityscapes/Waymo 일부 샘플이 **drivable/lane을 ‘없음’으로 학습**시켜, 대회 도메인에서 drivable/lane이 크게 줄어드는 방향으로 붕괴할 가능성이 큽니다.                                                                           | - DATASET_CONVERSION_SPEC **§5.5 RLMD Adapter**: drivable mask를 “all 0”로 둠<br>- DATASET_CONVERSION_SPEC **§5.2 Cityscapes Adapter / §5.4 Waymo Adapter**: lane 없으면 lane mask “0” 유지<br>- BDD100K paper p3 **§3**: “heterogeneous multitask learning” (태스크/라벨 이질성)                                                                                                              | **정확도/강건성 대폭↑** (특히 Drivable/Lane), 음성 편향 방지. Latency 영향 없음 | Med                                | **P0**              | PRD **7.2 표준 포맷**, **8.1 아키텍처 요구사항**(loss 계산 규칙), **10.1 정확도 게이트**(source별 지표)<br>Spec **3.7 Split Manifest Schema**, **5.* Adapter**, **8 Validation Rules**    |
| 2) **Split 누수 방지 강화: camera_id를 split group key에서 제거 + 소스별 split 규칙 고정(도시/시퀀스/공간 단위)**                      | 2카메라 런타임(동일 주행/동일 장면)에서 `camera_id`를 grouping에 포함하면, **cam0는 train, cam1은 val** 같은 누수가 가능합니다. 대회에서 성능이 무너지는 대표 패턴입니다. 또한 Cityscapes/KITTI-360/BDD100K는 본래 split 설계 철학이 달라, 이를 무시하면 일반화 평가가 왜곡됩니다.                                                        | - DATASET_CONVERSION_SPEC **§7 Split Policy**: grouping key `{source, sequence, camera_id}`<br>- PRD **7.4 데이터 품질 게이트**: “동일 원본 시퀀스의 train/val 중복 금지”<br>- Cityscapes paper p3 **§2.3**: “split at the city level”<br>- BDD100K paper p3 **§3**: video를 train/val/test로 분할(70k/10k/20k)<br>- KITTI-360 paper p23 **Appendix F.2 Dataset Split**: “without spatial overlapping” | **강건성↑(실전 일반화)**, 검증 신뢰도↑. Latency 영향 없음                    | Med                                | **P0**              | PRD **7.4 데이터 품질 게이트**, **10.1 정확도 게이트 (Validation Set)**, **11. 검증 시나리오**<br>Spec **7 Split Policy**, **3.3 Sample ID**, **3.7 Split Manifest Schema**          |
| 3) **PRD↔Spec 계약 정합: “학습용(da/lane binary) vs SPADE용(semantic_id)”를 문서/트리/토픽에서 일관화**                         | 지금 PRD는 segmentation을 “mono8 semantic id map”이라고 쓰면서(§7.2) 동시에 디렉터리는 da/lane 분리 형태(§7.3)이고, ROS2 출력 토픽도 semantic_id만 존재(§9.2)합니다. 구현자가 **어떤 마스크를 학습/배포에 쓰는지 혼선**이 생겨 QA/디버깅이 불가능해집니다.                                                                    | - PRD **7.2 표준 포맷** vs **7.3 디렉터리 규약**(semantic_id 디렉터리 부재)<br>- PRD **FR-02. 멀티태스크 출력** vs **9.2 출력 토픽**(da/lane 토픽 없음)<br>- DATASET_CONVERSION_SPEC **§3.2 Directory Layout**: `labels_semantic_id/` 포함<br>- DATASET_CONVERSION_SPEC **§3.6 Segmentation Mask Format**: da/lane + semantic_id 구성 규칙                                                                          | **강건성↑(통합/디버깅)**, 실수/재작업↓. Latency 영향 없음                    | Low                                | **P0**              | PRD **FR-02. 멀티태스크 출력**, **7.2 표준 포맷**, **7.3 디렉터리 규약**, **9.2 출력 토픽**<br>Spec **3.2 Directory Layout**, **3.6 Segmentation Mask Format**                        |
| 4) **Cityscapes lane 라벨 부재 처리: “0 채움” 금지, `ignore(255)`로 export + `has_lane=0`로 명시**                        | Cityscapes는 “road에 road markings 포함”이라고 정의하지만, lane marking을 별도 class로 제공하지 않습니다. 그런데 스펙은 `road_line`을 기대하고 없으면 lane mask=0로 둡니다. 결과적으로 lane이 있는 픽셀도 **모두 background로 학습**되어 lane head가 붕괴합니다.                                                           | - DATASET_CONVERSION_SPEC **§5.2 Cityscapes Adapter**: `road_line` 없으면 lane=0<br>- Cityscapes paper p19 **Table 8 (appendix)**: “road … including road markings” (차선표시가 road에 포함)                                                                                                                                                                                              | **Lane 정확도/강건성↑**(음성 라벨 오염 제거). Latency 영향 없음               | Med                                | **P0**              | Spec **5.2 Cityscapes Adapter**, **3.6 Segmentation Mask Format**, **3.7 Split Manifest Schema**, **8 Validation Rules**<br>PRD **7.1 데이터 소스**(Cityscapes 역할 명시) |
| 5) **RLMD 사용 정책 재정의: ‘모든 노면표시=lane’ 금지(또는 MVP에서는 RLMD 제외)**                                                 | RLMD는 road lines뿐 아니라 arrows/text 등 **다양한 노면표시(25 categories)**를 포함합니다. 이를 lane(차선)으로 전부 합치면, 대회 트랙의 화살표/문자/횡단보도 등이 lane으로 과검출되어 SPADE 입력이 오염될 수 있습니다. RLMD 세부 category→lane 경계선만 매핑이 불가하면, **MVP에서는 제외**가 안전합니다(unknown일 때는 제외가 “즉시 실행 가능한” 가장 안전한 수정). | - RLMD paper p1 **§I–II**: 25 categories, road lines + markings(화살표/텍스트 등 포함) 언급<br>- DATASET_CONVERSION_SPEC **§5.5 RLMD Adapter**: “All valid road marking pixels -> lane=1”                                                                                                                                                                                                 | **강건성↑(lane FP 감소)**, 대회 성능 안정화. Latency 영향 없음              | Low(제외) / Med(정교 매핑)               | **P0**              | PRD **7.1 데이터 소스**, **6.1 Segmentation 클래스 (MVP 고정)**(lane 의미 정의)<br>Spec **5.5 RLMD Adapter**, **7 Split Policy**(소스 on/off)                                    |
| 6) **Lane 마스크 두께/표현 표준화: 학습 시 dilation(예: 6–10px) + 평가 시도 동일 규칙**                                           | 960×540에서 lane은 매우 얇고 단절되기 쉬워 IoU가 불안정합니다. 라벨 두께/표현을 표준화하지 않으면 데이터셋마다 lane 분포가 달라 학습이 흔들립니다.                                                                                                                                                             | - YOLOPv2 p5 **§4.3.4**: 학습용 lane mask를 “width 8 pixels”로 생성(테스트는 2px)                                                                                                                                                                                                                                                                                                         | **Lane 정확도↑**, 학습 안정성↑. Latency 영향 없음(학습/GT 전처리)            | Low                                | **P0**              | PRD **10.1 정확도 게이트**(lane metric 정의), **11. 검증 시나리오**<br>Spec **3.6 Segmentation Mask Format**(lane mask 생성 규칙 옵션화)                                              |
| 7) **소형 객체 유지 전략: bbox min-area drop 제거/클래스 예외 + small-object 지표(AP_small/recall) 게이트 추가**                  | 변환 스펙의 “pixel area <16 drop”은 콘/볼라드/낙하물 같은 **대회 핵심 소형 장애물**을 학습/평가에서 날려버릴 수 있습니다(960×540에서 특히).                                                                                                                                                          | - DATASET_CONVERSION_SPEC **§3.5 Detection Label Format**: “Boxes with pixel area `< 16` are dropped”<br>- YOLO26 p4 **§2.3**: STAL(small-target-aware label assignment)로 small object 성능을 강조<br>- Cityscapes p5 **§3.1**: downsampling이 small objects(iIoU)에 큰 영향                                                                                                             | **OD(소형) 정확도↑**, 강건성↑. Latency 영향 없음                        | Low                                | **P0**              | PRD **10.1 정확도 게이트 (Validation Set)**(AP_small/클래스별 recall), **7.4 데이터 품질 게이트**(small-box 통계)<br>Spec **3.5 Detection Label Format**, **8 Validation Rules**     |
| 8) **멀티태스크 헤드 feature level 분리(DA는 shallow, Lane은 deep+deconv) 설계 명시 및 구현**                                 | DA는 깊은 feature가 반드시 필요하지 않은 반면 lane은 고수준/저수준 융합이 중요합니다. 같은 깊이에서 둘을 처리하면 멀티태스크 간섭과 수렴 난이도가 증가합니다.                                                                                                                                                         | - YOLOPv2 p3 **§3.2.2 Task Heads**: DA head를 FPN 이전에 연결(깊은 feature는 DA에 불필요+수렴 방해), Lane head는 FPN 끝+deconvolution 사용                                                                                                                                                                                                                                                          | **Seg 정확도↑**, 학습 안정성↑, 경우에 따라 Latency도↓(불필요 연산 감소)          | Med                                | **P0**              | PRD **8.1 아키텍처 요구사항**(head 연결 위치/decoder 스펙), **13. 리스크 및 대응**                                                                                                   |
| 9) **Lane/DA 손실 설계: (DA=CE) + (Lane=Focal+Dice) + 멀티태스크 loss balance 정책(예: ProgLoss)**                      | lane은 class imbalance가 극심하고(얇은 픽셀), 멀티태스크는 한 태스크가 다른 태스크를 눌러버리는 문제가 흔합니다. PRD는 “가중치 가능”만 있고 정책이 없습니다.                                                                                                                                                    | - YOLOPv2 p4 **§3.2.3**: DA=CrossEntropy, Lane=Focal(+Dice) 하이브리드 제시<br>- YOLO26 p4 **§2.3**: ProgLoss로 progressive loss balancing 언급                                                                                                                                                                                                                                          | **정확도↑(특히 Lane)**, 학습 안정성↑. Latency 영향 없음                   | Med                                | P1                  | PRD **8.1 아키텍처 요구사항**, **Stage C. Ablation and Tuning**                                                                                                          |
| 10) **조건 분해 Validation(야간/우천/역광/가림) + temporal flicker metric을 “정의”하고 게이트에 포함**                             | PRD는 시나리오 나열(§11)은 있지만, “어떤 지표/어떤 데이터로 합격인지”가 비어 있습니다. 대회는 연속 주행이므로 flicker가 치명적입니다.                                                                                                                                                                     | - PRD **11. 검증 시나리오**: 악조건/장시간/flicker 언급(정의 없음)<br>- BDD100K p3 **§3.1**: weather/time-of-day tags 제공(조건 분해 가능)<br>- Waymo PVPS p6: 카메라별 domain gap 및 temporal sampling 언급(시간축 평가 중요)                                                                                                                                                                                         | **강건성↑**, 조기 경보 가능. Latency 영향 없음                           | Low                                | **P0**              | PRD **10.1 정확도 게이트**, **11. 검증 시나리오**<br>Spec **3.7 Split Manifest Schema**(가능한 소스는 weather/time tag 포함, 불가하면 `unknown`)                                         |
| 11) **Profile(960×544 vs 768×448) 별 성능/지연 리포트 + profile 전환 시 품질 보장(멀티스케일 또는 2-stage fine-tune)**            | PRD는 profile 전환을 런타임에 허용(§8.3)하지만, 학습/평가가 profile별로 어떻게 보장되는지 없습니다. 작은 해상도는 lane/소형 객체에 치명적입니다.                                                                                                                                                          | - PRD **8.3 추론 프로파일**: quality/latency profile 정의<br>- YOLOPv2 p4 **§4.2**: train/test에서 해상도 전략을 분리<br>- Cityscapes p5 **§3.1**: downsampling이 small objects에 악영향                                                                                                                                                                                                              | **Latency/Accuracy trade-off 예측 가능**, 운영 안정성↑               | Med                                | P1                  | PRD **8.2 입력 전처리**, **8.3 추론 프로파일**, **10.2 런타임 게이트**                                                                                                            |
| 12) **배포/모니터링 방법론: TensorRT(최소 FP16) 경로를 “즉시” 구체화 + 런타임 KPI 계측(semantic 위반, FPS, p95 latency, drop ratio)** | RTX4060에서 2카메라×30FPS는 여유가 크지 않을 수 있습니다(특히 멀티헤드+후처리). PRD의 런타임 게이트(§10.2)를 만족하려면 계측/경보가 먼저 필요합니다.                                                                                                                                                         | - PRD **3.1 배포/런타임**, **10.2 런타임 게이트**: p95 latency/drop ratio 요구<br>- YOLO26 p6 **§4.1/§4.2**: ONNX/TensorRT export 및 FP16/INT8 quantization 언급                                                                                                                                                                                                                               | **Latency/Robustness↑**, 실전 장애 조기 탐지                        | Med                                | **P0**              | PRD **10.2 런타임 게이트**, **9.4 QoS 정책**(계측 포함), **14. 산출물**(벤치 리포트)<br>Spec **8 Validation Rules**(런타임에서도 동일 위반 체크)                                                 |

---

B) Red Flags (max 8)
(요청하신 “대회에서 성능이 무너질 가능성이 큰 시나리오 5개”만 선정했고, 각 항목에 최소 실험 1개를 포함했습니다.)

1. **Failure risk:** 야간/우천/역광(헤드라이트/글레어)에서 **Lane/Drivable segmentation 붕괴** → SPADE 입력 품질 급락

   * **Early warning metric:**

     * 조건 분해 val에서 `Lane IoU(야간/우천)` 급락(예: <0.35)
     * `lane connected components` 수 급증(단절/노이즈)
     * `semantic flicker rate`(연속 프레임 class 변화율) 급증
   * **Mitigation:**

     * (데이터) 자가 수집으로 야간/우천/역광 최소 세트 구성 + BDD100K tags 기반 조건 val 구축
     * (학습) photometric/weather augmentation(노이즈/블러/감마/가짜 빗방울) 추가
     * **최소 실험:** “baseline vs weather/photometric aug” 2-run 학습 후 `야간/우천 subset`에서 Lane IoU + flicker 비교

2. **Failure risk:** 원거리 소형 장애물(traffic_cone/bollard/road_obstacle) **미검출** → 대회 주행에서 충돌/경로이탈

   * **Early warning metric:**

     * 클래스별 `recall@{거리/픽셀크기}`(예: bbox height<12px 구간)
     * `AP_small`(또는 small bucket mAP50)
   * **Mitigation:**

     * 변환 단계의 `bbox area < 16 drop` 제거(또는 cone/bollard 예외)
     * small-object 강화(STAL 유사 샘플링/label assignment)
     * **최소 실험:** “drop<16 ON/OFF” 변환→동일 epoch 학습→ `traffic_cone/bollard AP_small` 비교

3. **Failure risk:** 부분 라벨 소스(RLMD/Cityscapes/Waymo 일부)가 drivable/lane을 **‘없음(0)’으로 오염** → drivable이 줄고 lane이 사라짐

   * **Early warning metric:**

     * 소스 추가 전/후 `Drivable mIoU`, `Lane IoU`가 동시에 하락
     * drivable/lane positive pixel ratio가 비정상 방향으로 이동(예: drivable 과소)
   * **Mitigation:**

     * `has_da/has_lane/has_det`를 manifest에 기록하고, 없는 태스크는 `ignore(255)`로 export + loss masking
     * **최소 실험:** 동일 데이터 혼합에서 “loss masking OFF(0채움)” vs “loss masking ON(ignore)” 비교

4. **Failure risk:** RLMD/자체 데이터의 **노면표시(화살표/문자/횡단보도)**가 lane으로 합쳐져 **lane FP 폭증** → SPADE가 lane을 구조선으로 오인

   * **Early warning metric:**

     * “화살표/문자/횡단보도 포함 프레임” subset에서 `lane FP rate` 급증
     * conversion warning인 `Lane positive pixel ratio > 0.35` 빈번
   * **Mitigation:**

     * (가능하면) RLMD category 중 “차선 경계선”만 lane으로 매핑, 나머지는 background
     * (불가능/unknown이면) **MVP에서는 RLMD를 lane 학습에서 제외**
     * **최소 실험:** “RLMD 포함 vs 제외” 학습 2-run 후, 자체 수집(화살표/횡단보도) subset에서 lane FP 비교

5. **Failure risk:** split 누수(동일 주행/장면이 train과 val에 동시 존재)로 **검증 성능 과대평가** → 대회에서 급락

   * **Early warning metric:**

     * train↔val 간 **near-duplicate 탐지**(perceptual hash/embedding nearest neighbor) 비율 상승
     * val은 높은데 “대회 유사 holdout(self-collected)” 성능이 급락하는 괴리
   * **Mitigation:**

     * grouping key에서 `camera_id` 제거 + 소스별 split 정책(도시/시퀀스/공간) 고정
     * **최소 실험:** 현재 split과 수정 split 각각에서 학습 후, “대회 유사 holdout” 성능 차이 비교(괴리 감소 확인)

---

C) Action Plan

### MVP now (Next 2 weeks, day-by-day checklist)

**Day 1 — 문서/계약 즉시 정리(구현 혼선 제거)**

* [ ] PRD **7.2/7.3**에 “학습용(da/lane binary) + 배포용(semantic_id)”를 명확히 분리해 문장/디렉터리 정합
* [ ] PRD **FR-02**, **9.2 출력 토픽**: (선택지 A) da/lane은 내부 출력으로만 두고 문구 수정 / (선택지 B) debug 토픽 추가를 명시(둘 중 하나로 고정)
* [ ] Spec **7 Split Policy**에서 grouping key를 `{source, sequence}`로 수정(※ camera_id 제거)
* [ ] Spec **5.2 Cityscapes Adapter**: lane은 `ignore=255`, `has_lane=0`로 고정(road_line 기대 제거)
* [ ] Spec **5.5 RLMD Adapter**: MVP에서는 “RLMD lane 학습 제외”를 기본값으로 명시(세부 매핑은 unknown이므로 V2로 이관)
* 산출물: PRD v0.4, Spec v1.1 커밋(섹션 변경 내역 포함)

**Day 2 — 변환 파이프라인 핵심 수정(부분 라벨/누수 방지)**

* [ ] `split_manifest.csv`에 `has_det, has_da, has_lane` 컬럼 추가
* [ ] 마스크 export 규칙: 라벨 없는 태스크는 **0이 아닌 255(ignore)** 로 저장(da/lane 각각)
* [ ] QA validator에 “ignore 값 허용” 규칙 추가(semantic_id는 여전히 {0,1,2}만)
* [ ] split 생성 로직: sequence 단위 그룹핑이 실제로 동작하도록 source별 `sequence` 정의 규칙을 코드에 명시(예: bdd=video_id, cityscapes=city, self=run_id)
* 산출물: converter PR + 단위 테스트(샘플 20개로 hard-fail 0 보장)

**Day 3 — 소스별 최소 변환 샘플셋 생성 + QA 리포트**

* [ ] BDD100K 200장, Cityscapes 200장, self-collected 200장(가능하면) 변환 실행
* [ ] `conversion_report.json`, `source_stats.csv`, `checksums.sha256` 생성 확인
* [ ] warning top-10을 사람이 리뷰하고 “허용/수정” 판정 기록
* 산출물: `datasets/pv26_v1_small/` + QA 리포트

**Day 4 — 학습 데이터로더/손실 마스킹 구현**

* [ ] dataloader에서 `has_*`에 따라 task별 loss를 0으로 마스킹(ignore 픽셀은 loss 제외)
* [ ] Lane mask dilation(학습 GT 전처리) 옵션 추가(기본 8px 폭 타깃)
* [ ] metric 코드에 “조건별 subset 평가” 훅 추가(최소 day/night 분리)
* 산출물: 학습 1epoch smoke test(손실 감소 + NaN 없음)

**Day 5 — 모델 구조 1차 고정(멀티헤드 분리)**

* [ ] YOLO26 backbone + 3 heads(OD/DA/Lane) 구성
* [ ] YOLOPv2 방식으로 DA head는 shallow tap, Lane head는 deep tap(+upsample/deconv)로 연결
* [ ] Loss: DA=CE, Lane=Focal(+Dice 옵션), OD=기존 YOLO loss (가중치 고정값 1차 설정)
* 산출물: forward/backward 정상 + torchscript/onnx export 사전 점검(unsupported op 목록 기록)

**Day 6 — Baseline 학습(가장 깨끗한 소스 중심)**

* [ ] BDD100K 중심으로 baseline 50~100epoch(짧게) 학습
* [ ] Val: day/night 분리 + lane/drivable 지표 기록
* 산출물: baseline weight + 리포트(OD mAP50, DA mIoU, Lane IoU)

**Day 7 — Self-collected holdout(대회 유사) 최소 구축**

* [ ] self-collected 1) train 2) holdout(대회 유사)로 분리(시퀀스 단위)
* [ ] holdout에서 baseline 성능 측정(이 수치가 “실전 기준”)
* 산출물: “public-val vs self-holdout” 괴리 리포트

**Day 8 — Red Flag 실험 1~2번(야간/우천 + 소형 객체)**

* [ ] 실험 A: photometric/weather aug ON/OFF 비교(동일 epoch)
* [ ] 실험 B: bbox min-area drop ON/OFF 비교(동일 epoch)
* 산출물: 2-run 비교표(조건별 Lane IoU, cone/bollard recall)

**Day 9 — ROS2 노드 MVP 구현(정확한 계약/타임스탬프)**

* [ ] 입력: `/yolopv26/cam0/image_raw`, `/yolopv26/cam1/image_raw`
* [ ] 출력: `/yolopv26/cam*/semantic_id`(mono8), `/yolopv26/cam*/detections`
* [ ] semantic_id 값 검증(0/1/2 외 값이면 clamp+카운트 증가)
* [ ] 타임스탬프 보존 및 mismatch 측정 로깅
* 산출물: rosbag 재생으로 end-to-end 동작 확인

**Day 10 — 성능 계측/모니터링(게이트를 “측정 가능”하게)**

* [ ] FPS, p95 latency, dropped frame ratio, GPU mem 사용량을 주기적으로 출력/로그
* [ ] 30분 연속 실행 테스트 스크립트 작성
* 산출물: PRD §10.2 지표가 로그로 재현됨

**Day 11 — ONNX/TensorRT 경로 최소 확보(FP16 목표)**

* [ ] ONNX export(고정 input: 960x544, batch=1/2) 성공
* [ ] TensorRT FP16 엔진 빌드 + 단일 프레임 결과 일치(정확히 일치가 어려우면 허용 오차 정의)
* 산출물: RTX4060에서 cam당 30FPS 달성 여부 1차 판정

**Day 12 — Red Flag 실험 3~5번(부분 라벨/노면표시/누수)**

* [ ] 실험 C: loss masking OFF(0채움) vs ON(ignore) 비교
* [ ] 실험 D: RLMD 포함 vs 제외 비교(또는 self road-marking subset으로 FP 측정)
* [ ] 실험 E: split policy 수정 전/후로 self-holdout 성능 괴리 비교
* 산출물: 3-run 비교표 + 결론(채택 설정 고정)

**Day 13 — RC 재학습 + 프로파일(960x544 vs 768x448) 리포트**

* [ ] 채택 설정으로 RC 학습(더 긴 epoch)
* [ ] 두 profile에서 정확도/latency 리포트 생성
* 산출물: `best/rc` weight + profile별 리포트

**Day 14 — SPADE 연동 최종 검증 + Freeze**

* [ ] `/spade/input/semantic_id` 리다이렉트/직접 발행 동작 확인
* [ ] semantic_id 계약 위반 0, timestamp mismatch <33ms 확인
* [ ] PRD Stage E 완료 조건 체크 후 RC 태깅
* 산출물: 릴리스 후보 패키지(모델+노드+리포트+class_map)

### V2 later (MVP 이후)

* [ ] RLMD 25-class의 “차선 경계선” subset만 선별 매핑(세부 매핑은 현재 문서 기준 **unknown**)
* [ ] Temporal 안정화(간단한 EMA/majority vote 또는 lightweight temporal head)로 flicker 추가 저감
* [ ] INT8(QAT 포함) 및 dynamic shape TensorRT 최적화
* [ ] 멀티카메라 간 일관성(동일 장면의 cross-camera consistency) 지표/보정 로직
* [ ] obstacle pixel-wise segmentation(V2 backlog) 및 road-marking multi-class 확장(계약 변경 필요 시 별도 RFC)
