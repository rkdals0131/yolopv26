A) Top Missing Methodologies (max 12)

| Methodology                                                                          | Why it matters for our exact setup                                                                                                                                                                                   | Evidence from which paper/doc (with page/section)                                                                                                                     | Expected impact (accuracy/latency/robustness)    | Implementation cost | Priority | Where to update in PRD/spec (exact section names)                                                                                             |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 1) **Lane rasterization 표준화** (polyline/2-line → mask, train/eval thickness 규정)      | Lane은 “가늘고 끊기기 쉬운” 타스크라서 **마스크 두께/래스터화 규칙이 IoU를 사실상 결정**합니다. PRD는 Lane IoU 게이트만 있고(정의 불명확), Conversion spec은 “Lane categories → lane=1”만 있고 **두께/중심선 규칙이 없음**. 이 상태면 팀 내/실험 간 지표가 흔들리고, 대회 환경에서 lane이 쉽게 붕괴합니다.      | YOLOPv2는 BDD100K lane이 “두 줄”로 주어져 **centerline 계산 후 train에서는 8px, test는 2px**로 그린다고 명시(§4.3.4, p.5).                                                                  | Accuracy↑ (lane IoU 재현성), Robustness↑            | Low                 | P0       | PRD: `10.1 정확도 게이트 (Validation Set)`, `7.2 표준 포맷` / Spec: `5.1 BDD100K Adapter`, `3.6 Segmentation Mask Format`                               |
| 2) **semantic_id 생성/파일 존재 규칙을 PRD↔Spec 단일화** (has_semantic_id와 누락검사 정의 포함)           | PRD는 `labels_semantic_id/`를 디렉터리 규약에 “항상 존재”처럼 두고, Spec은 `has_semantic_id=1`일 때만 semantic mask를 요구합니다. 지금대로면 “라벨 누락 검사 100%” 같은 PRD 게이트(문구)가 **부분 라벨 정책과 충돌**할 소지가 큽니다(특히 CI/QA에서).                                  | Conversion spec: `labels_semantic_id`는 `has_semantic_id=1`일 때만 필요, `255` 금지, 합성 순서(§3.6, §3.7). PRD: semantic 출력 계약(§FR-02/03), 표준 포맷/디렉터리 규약(§7.2/7.3).              | Robustness↑ (파이프라인/QA 일관성), Dev velocity↑        | Low                 | P0       | PRD: `FR-02. 멀티태스크 출력`, `7.4 데이터 품질 게이트` / Spec: `3.6 Segmentation Mask Format`, `8.1 Hard Fail Rules`, `3.7 Split Manifest Schema`           |
| 3) **Detection “부분 라벨” 방법론 추가** (per-source/per-class coverage + loss masking)       | 현재 PRD의 부분 라벨 정책은 **Seg 중심**이고, Spec도 `has_det`만 있습니다. 그런데 Cityscapes/일부 소스는 **탐지 클래스 커버리지가 제한**될 가능성이 커서(예: 인간/차량 위주), 이를 “라벨 없음=배경”으로 학습하면 **거짓 음성(negative) 학습**이 누적되어 OD 성능이 붕괴합니다(특히 road_obstacle/sign_pole류). | Cityscapes는 **humans/vehicles에 instance-level 라벨 제공**을 강조(§2.2, p.2).  / PRD는 partial label을 `ignore(255)+has_*`로 강제하지만 detection에 대한 per-class 범위 정의가 없음(§FR-05).    | Accuracy↑ (OD), Robustness↑ (도메인 혼합)             | Med                 | P0       | PRD: `FR-05. 부분 라벨(Partial Labels) 처리 정책`, `8.1 아키텍처 요구사항` / Spec: `3.7 Split Manifest Schema`, `5.2 Cityscapes Adapter`, `8.2 Warning Rules` |
| 4) **“small box drop(<16px)” 정책 재설계** (class-aware/옵션화 + small-object 학습 강화)         | Spec의 `pixel area <16 drop`은 cone/bollard 같은 **소형 물체를 구조적으로 제거**할 위험이 큽니다(우리 det class에 소형이 많음). 대회에서 멀리 있는 cone/bollard/낙하물 검출이 무너지면 치명적입니다.                                                                        | Cityscapes는 다운샘플링이 iIoU(소형 객체 민감 지표)를 크게 악화시킨다고 명시(§3.2, p.5).  / YOLO26은 **STAL(small-target-aware label assignment)**로 small/occluded object를 강화한다고 설명(§2.3, p.4).  | Accuracy↑ (특히 small objects), Robustness↑        | Low~Med             | P0       | PRD: `10.1 정확도 게이트`, `7.4 데이터 품질 게이트` / Spec: `3.5 Detection Label Format`, `8.2 Warning Rules`                                               |
| 5) **멀티태스크 loss 결합 “구체값/스케줄” 고정** (기본 loss set + weight schedule)                    | PRD는 “가중치 설정 가능”만 있고, 실제로는 **task 간 gradient 간섭**이 가장 먼저 문제를 만들 가능성이 큽니다(Drivable이 너무 쉬워 backbone을 지배하거나, Lane이 희소해서 죽는 케이스). “가능”이 아니라 **기본값(초기 weight, 스케줄, 로깅)**을 문서에 고정해야 재현됩니다.                                 | YOLOPv2는 detection loss를 class/obj/box 가중합으로 두고(§3.2.3, p.4) , YOLO26은 **ProgLoss(진행형 loss balancing)**를 loss 안정화로 제시(§2.3, p.4).                                     | Accuracy↑, Robustness↑ (수렴/간섭)                   | Med                 | P0       | PRD: `8.1 아키텍처 요구사항`(하위에 “8.x Loss/Optimization” 신설 권장), `12. 개발 단계` / Spec: (필수 아님)                                                          |
| 6) **Seg loss 선택을 “태스크별로 명시”** (lane=Focal+Dice, drivable=CE/BCE + ignore)           | Lane은 희소/가늘어 CE만 쓰면 금방 background로 붕괴합니다. PRD는 ignore/has 마스킹만 있고 **어떤 loss를 쓰는지 고정이 없음**.                                                                                                                           | YOLOPv2는 drivable에 CE, lane에 focal, 그리고 **dice+focal hybrid**를 실험/채택(§3.2.3, p.4; ablation Table 5도 동일 방향, p.5).                                                      | Accuracy↑ (lane/drivable), Robustness↑           | Low                 | P0       | PRD: `8.1 아키텍처 요구사항`(loss 명시), `10.1 정확도 게이트`(metric 정의와 연동)                                                                                  |
| 7) **학습 augmentation 정책을 “결정값”으로 추가** (mosaic/mixup + photometric, mask 동기변환)        | 대회 도메인 갭을 메우려면 augmentation이 사실상 필수입니다. 지금 Spec은 “training augmentation out-of-scope”인데, PRD에도 “정책/기본값”이 없습니다. 멀티태스크는 augmentation이 조금만 틀어져도 seg mask와 det bbox가 어긋나 품질이 즉시 무너집니다.                                   | YOLOPv2는 Mosaic/Mixup을 multi-task에서 성능 향상 BoF로 강조(§2.4~§3.2.3, p.2~4; ablation에도 반영 p.5).                                                                             | Robustness↑, Accuracy↑                           | Low                 | P1       | PRD: `8. 모델/학습 요구사항`(“8.x Augmentation” 신설), `12. Stage C`                                                                                    |
| 8) **Train resize/multi-scale 정책을 런타임 프로파일과 정합** (960x544/768x448 포함)                | Spec은 “원본 해상도 유지”라서 소스별 해상도가 섞입니다. PRD는 런타임에 960x544(letterbox)와 768x448 프로파일을 고정해 두었습니다. 학습이 이 입력분포를 커버하지 않으면 **프로파일 전환 시 성능 급락**이 쉽게 발생합니다.                                                                        | YOLOPv2는 train/test에서 해상도를 다르게 리사이즈하는 프로토콜을 명시(§4.2, p.4).  / Cityscapes는 다운샘플링이 small object 지표를 악화시킨다고 명시(§3.2, p.5).                                               | Robustness↑, Accuracy↑ (profile 전환), Latency 안정↑ | Med                 | P0       | PRD: `8.2 입력 전처리`, `8.3 추론 프로파일`, `10.2 런타임 게이트` / Spec: `3.4 Image Requirements`(“conversion은 유지, training은 profile-resize” 문구 추가 권장)        |
| 9) **2카메라 도메인 갭 대응** (camera_id-conditioned BN/정규화 + per-camera val 리포트)             | 실제 시스템은 2카메라이고, 카메라마다 FOV/노출/왜곡/위치가 달라 분포가 갈립니다. 단일 모델/단일 정규화로 밀면 “한 카메라만 잘 되고 다른 카메라가 무너지는” 케이스가 흔합니다. PRD는 멀티카메라 입출력/동기만 있고 **학습·검증의 per-camera 정책이 없음**.                                                          | Waymo PVPS는 카메라마다 클래스 분포가 다르고, **단일 카메라로 학습한 모델이 다른 카메라로 일반화되지 않을 수 있다(큰 domain gap)**고 명시(§(Fig.3 주변) p.6).                                                          | Robustness↑ (2카메라), Accuracy↑                    | Med                 | P1       | PRD: `10.1 정확도 게이트`(per-camera 게이트 추가), `11. 검증 시나리오`, `FR-01` / Spec: `3.7 Split Manifest Schema`(camera_id 활용을 리포트에 연결)                     |
| 10) **날씨/시간대 stratified split + 조건별 합격게이트** (manifest tag를 실제로 사용)                   | PRD는 악조건(야간/우천/역광/가림) 검증을 요구하지만, Spec은 weather/time tag를 기록만 하고 **split/val 구성에 강제 조건이 없습니다**. Cityscapes 자체는 악천후가 거의 없으므로(=소스 편향), stratification을 안 하면 val이 “맑은 낮” 위주로 고정되고 대회에서 폭락합니다.                            | Cityscapes는 **폭우/눈 같은 악천후를 의도적으로 수집하지 않았다**고 명시(§2.1, p.2).                                                                                                           | Robustness↑ (대회 환경), Accuracy↑(조건별)              | Low                 | P0       | PRD: `10.1 정확도 게이트`, `11. 검증 시나리오` / Spec: `7. Split Policy`, `3.7 Split Manifest Schema`                                                     |
| 11) **배포/가속 방법론을 “정식 절차”로 고정** (ONNX→TensorRT FP16, static shape, NMS 전략)            | 목표 런타임이 2카메라 60fps라서, “단계적 전환 허용”만으로는 부족합니다. **TensorRT 변환·정확도 회귀·latency 측정·shape 정책**을 문서/CI로 고정하지 않으면 통합 단계(Stage D/E)에서 지연/불안정이 터집니다.                                                                            | YOLO26은 **NMS-free가 NMS 지연/튜닝 의존을 제거**해 배포 파이프라인을 단순화한다고 설명(§2.2, p.4).  또한 FP16/INT8 양자화에서의 효율을 별도 섹션으로 강조(§4.2, p.6).                                               | Latency↓, Robustness↑ (배포 재현성)                   | Med~High            | P0       | PRD: `3.1 배포/런타임 고정 조건`, `8.3 추론 프로파일`, `10.2 런타임 게이트`                                                                                        |
| 12) **후처리/모니터링 방법론 추가** (threshold, morphology, temporal smoothing, anomaly metrics) | PRD는 “class flicker 측정”만 있고 **줄이는 방법(후처리/temporal smoothing)이 비어 있습니다**. 또한 runtime에서 semantic_id가 깨지거나(0/1/2 외 값), lane/drivable 픽셀 비율이 튀면 SPADE가 흔들릴 수 있는데, 이를 조기 탐지하는 지표가 PRD에 고정돼 있지 않습니다.                       | YOLOPv2는 시각적 비교에서 **discontinuous lane prediction**, false/missing drivable 등을 구체적으로 언급(§4.3.5, p.5).  / Spec은 lane pixel ratio 같은 이상치 경고 규칙을 이미 갖고 있음(§8.2).         | Robustness↑ (flicker/이상치), Debuggability↑        | Low~Med             | P0       | PRD: `11. 검증 시나리오`, `10.2 런타임 게이트`, `9. ROS2 인터페이스 계약`(debug topic 기준), `13. 리스크 및 대응` / Spec: `8.2 Warning Rules`(runtime 모니터링으로 확장 문구)      |

---

B) Red Flags (max 8)
(요청하신 “대회에서 성능이 무너질 가능성이 큰 시나리오 5개”를 그대로 Red Flag로 정리했습니다. 각 항목에 최소 실험 1개 포함)

1. **Failure risk:** 야간 + 젖은 노면(반사) + 헤드라이트/가로등 glare에서 **Lane이 끊기거나 Drivable이 번짐**

   * **Early warning metric:** (a) 야간/우천 tag subset에서 Lane IoU 급락, (b) lane connected components 수 급증(끊김), (c) frame-to-frame lane 픽셀 IoU(=flicker) p95 악화
   * **Mitigation:** lane loss를 Focal+Dice로 고정하고(방법론 6), lane rasterization 두께/평가 기준을 확정(방법론 1), 야간/우천 photometric augmentation(감마/노이즈/글레어) 추가(방법론 7).

     * **최소 실험:** `weather_tag=rain OR time_tag=night` subset을 따로 val로 고정하고, (i) baseline vs (ii) 감마/글레어 augmentation 추가 vs (iii) lane loss Dice 추가 3-way 비교. 성공 기준: subset Lane IoU + flicker 지표 동시 개선.

2. **Failure risk:** 역광/터널 출입구/HDR에서 **Drivable이 과대/과소 분할**(노면-인도 경계 붕괴)

   * **Early warning metric:** drivable 픽셀 비율의 분산 급증(특히 밝기 상위 5% 프레임), drivable mIoU의 조건별 급락
   * **Mitigation:** drivable head 연결 위치/upsampling 전략을 명시하고(방법론 5/8), 밝기/콘트라스트/색온도 augmentation을 “결정값”으로 고정(방법론 7), 후처리로 작은 hole 제거/경계 smoothing(방법론 12).

     * **최소 실험:** self-collected에서 역광/터널 프레임 300장만 수집해 “HDR subset” 생성 → (i) augmentation 없음 vs (ii) photometric augmentation 있음 비교. 성공 기준: subset drivable mIoU + drivable 픽셀 비율 안정화.

3. **Failure risk:** 공사구간/임시차선/다양한 노면표시에서 **Lane false positive 폭증**(특히 화살표/문자/횡단보도)

   * **Early warning metric:** lane positive pixel ratio가 Spec 경고 기준(>0.35) 초과 빈도 증가, lane FP area가 횡단보도/문자 영역에 집중(간단히 connected component의 형태/면적 분포로 감지)
   * **Mitigation:** RLMD는 25종 road marking(화살표/stop line/횡단보도 등 포함)임을 고려해 lane에 무작정 합치지 말고(방법론 1, 12), self-collected에서 공사구간을 별도 소스로 확보 + 조건별 게이트로 강제(방법론 10). RLMD를 쓰려면 “lane-boundary-equivalent”만 선별하는 mapping table을 먼저 버전관리(방법론 12와 연동).

     * **최소 실험:** 공사구간/횡단보도/화살표 포함 프레임 200장 curated set 생성 → (i) 현재 lane 모델의 lane pixel ratio/FP를 측정하고, (ii) 간단 morphology(작은 blob 제거 + thin/width 제어)를 적용한 후처리만으로 FP 감소량 측정. RLMD는 V2로 두되, “후처리만으로 줄일 수 있는 FP”인지 먼저 확인.

4. **Failure risk:** cone/bollard/낙하물 같은 **소형 장애물 OD가 누락**(원거리·부분가림에서 급락)

   * **Early warning metric:** small-object recall(예: bbox 면적 하위 20% 구간에서 recall@0.5) 급락, 특정 클래스(traffic_cone/bollard/road_obstacle) 예측 빈도 0에 수렴
   * **Mitigation:** Spec의 `<16px drop`을 즉시 옵션화/기본 비활성화(방법론 4), YOLO26의 STAL/소형 우선 할당 아이디어를 최소한 “small-object oversampling + label assignment 튜닝” 형태로 도입(방법론 4/5).

     * **최소 실험:** 동일 학습 설정에서 `min_box_area_px`를 (i)16 유지 vs (ii)0(드롭 없음)으로만 바꿔 1epoch 짧은 학습 후, small-object recall 비교. 차이가 크면 P0로 고정 변경.

5. **Failure risk:** 2카메라 동기 오차/드롭 + ROS2 backpressure로 **timestamp mismatch / stale frame**이 누적되어 SPADE 입력이 흔들림

   * **Early warning metric:** PRD 게이트 그대로: semantic timestamp mismatch(>33ms) 비율, dropped frame ratio, end-to-end latency p95(>50ms)
   * **Mitigation:** TensorRT FP16 + static shape 프로파일 고정(방법론 11), camera별 독립 worker + 최신 프레임 우선 드롭을 실제 코드로 강제(이미 PRD QoS 방향은 있음), runtime 모니터링에 “mismatch/dropped/invalid-id”를 상시 로깅(방법론 12).

     * **최소 실험:** rosbag(또는 녹화 입력)로 60fps 등가 부하를 30분 재생하며 (i) PyTorch, (ii) ONNXRuntime, (iii) TensorRT FP16 순서로 p95 latency/mismatch/dropped를 측정해 병목을 규명. 성공 기준: p95≤50ms, mismatch<33ms, drop<1%.

---

C) Action Plan

### MVP now (Next 2 weeks, day-by-day checklist)

**Day 1 — 문서 정합 “즉시 Fix”**

* PRD에 아래를 **명시적으로 추가/수정**(커밋 1개로 묶기)

  * `FR-05`에 detection partial-label(coverage) 항목 추가(방법론 3)
  * `7.4 데이터 품질 게이트`에 `has_semantic_id=0`인 경우 semantic 파일 “미존재 허용” 문구 추가(방법론 2)
  * `10.1 정확도 게이트`에 Lane metric 정의(두께/래스터화 기준 포함) 추가(방법론 1)
* Conversion spec 수정

  * `3.5 Detection Label Format`의 `<16px drop`을 `min_box_area_px` 파라미터로 변경, 기본값 0(드롭 없음)으로 설정(방법론 4)
  * `5.1 BDD100K Adapter`에 lane rasterization 규칙(센터라인/두께 px/anti-aliasing 여부)을 결정값으로 추가(방법론 1)
* 산출물: PRD v0.4→v0.5, Spec v1.1→v1.2, 변경점 로그

**Day 2 — Converter skeleton 구현(디렉터리/manifest/QA)**

* `tools/convert_dataset.py` 최소 기능 구현

  * output tree 생성(§3.2), sample_id 생성(§3.3)
  * `split_manifest.csv` 작성(§3.7)
  * `has_*` 플래그 적용(§3.6)
* `tools/validate_dataset.py` 구현

  * Spec §8.1 Hard Fail Rule 전부 구현(최소 1:1 매핑)
* 산출물: “raw 없는 더미 입력”으로도 변환/검증이 끝까지 도는 e2e 테스트

**Day 3 — BDD100K adapter(우선순위 1) + lane rasterization**

* BDD100K adapter 구현

  * det 변환 + bbox clip + min_box_area_px 적용
  * drivable mask 생성
  * lane polyline/2-line → centerline → thickness(결정값)로 mask 생성
  * `has_semantic_id = (has_da==1 && has_lane==1)` 규칙 구현 후 semantic_id 합성
* QA 통과(하드 fail=0) 확인
* 산출물: BDD100K 1,000샘플 conversion + stats 리포트(클래스 카운트/픽셀비율)

**Day 4 — Cityscapes adapter + detection coverage 메타데이터**

* Cityscapes adapter 구현

  * drivable: road+parking만 1
  * lane: 전 픽셀 255 + has_lane=0
  * detection: instance 기반으로 가능한 클래스만 생성(사람/차량 등)
* manifest에 `det_label_scope`(예: `full|subset`) 또는 `det_annotated_class_ids` 컬럼 추가(방법론 3)

  * training에서 subset이면 “해당 소스에 없는 클래스에 대해 loss를 걸지 않도록” 설계 문서화
* 산출물: Cityscapes conversion + QA + coverage 리포트

**Day 5 — Self-collected adapter 계약 고정 + annotation QA 규칙 실행**

* self-collected input 스키마 확정(파일명/카메라 ID/cam0·cam1 매핑)
* “hazard ROI 미라벨 샘플 invalid” 규칙을 **검증기로 구현**(sample QA 게이트)
* 산출물: self-collected 200샘플 conversion + QA

**Day 6 — Trainer 데이터로더/마스킹(부분 라벨) 구현**

* dataloader가 `split_manifest.csv` 기반으로 샘플 로드
* seg: ignore(255) + has_da/has_lane 마스킹 적용(이미 PRD 있음 → 코드로 확정)
* det: `det_label_scope` 반영(부분 커버리지 시 loss masking 또는 class-drop) 구현(방법론 3)
* 산출물: 1 step train이 오류 없이 돌아가고, loss가 task별로 정상 출력

**Day 7 — 모델/손실 baseline 확정(결정값)**

* YOLO26 backbone + 3 head 연결 설계 고정

  * drivable head는 얕은 feature에서(필요 시) / lane head는 깊은 feature에서(방법론 5/8의 설계 의도 반영)
* loss 고정:

  * det: 기본 YOLO loss
  * drivable: CE/BCE(+ignore)
  * lane: focal + dice(hybrid)
  * multi-task weight 초기값 + ProgLoss 스타일 스케줄(간단 버전이라도) 확정
* 산출물: config 1개로 재현되는 baseline 학습 설정

**Day 8 — 학습 입력 크기/프로파일 정합(960x544 + 768x448)**

* training preprocess에 letterbox 정책을 명시적으로 적용

  * 기본: 960x544
  * multi-scale 옵션: {960x544, 768x448}에서 샘플링
* 평가도 두 프로파일로 모두 수행하도록 evaluator 구현
* 산출물: 프로파일 전환 시 성능 급락 여부를 “수치로” 확인하는 리포트

**Day 9 — 첫 통합 학습(빠른 러닝) + 조건별 리포트**

* 10~20k 샘플로 1차 학습(짧은 epoch)
* 리포트는 반드시 breakdown 포함:

  * source별, weather_tag/time_tag별, camera_id별(가능한 범위)
* 산출물: PRD `10.1` 형식으로 “게이트 대비” 표 1장

**Day 10 — ONNX export + TensorRT FP16(quality/latency 프로파일 2개)**

* 두 입력 shape(960x544, 768x448) 각각에 대해

  * ONNX export → TRT engine build
  * 정확도 회귀(OD/DA/Lane 최소한의 샘플 검증)
  * latency 측정(ms, fps)
* 산출물: 엔진 파일 2개 + latency 리포트 + 회귀 테스트 스크립트

**Day 11 — ROS2 노드 통합 + semantic_id 발행 안정화**

* ROS2 Humble 노드로

  * cam0/cam1 subscribe → per-camera inference → publish semantic_id + detections
  * timestamp 보존/검증(33ms) 로직 구현
* 산출물: PRD `10.3 SPADE 연동 게이트`를 체크하는 자동 테스트(rosbag 기반)

**Day 12 — 후처리/모니터링(필수 지표) 추가**

* semantic_id invalid 값(0/1/2 외) 카운트
* lane/drivable 픽셀 비율, flicker 지표(frame-to-frame IoU) 로깅
* dropped frame ratio, p95 latency, GPU mem 사용량 로깅
* 산출물: 30분 런 “대시보드용 CSV/JSON” 자동 생성

**Day 13 — Red Flag 5개 최소 실험 실행(짧게라도)**

* B 섹션의 5개 시나리오에 대해, 최소 실험을 모두 수행하고 결과를 한 장으로 요약
* 산출물: “실험-결론-다음 액션” 1페이지 x5 (총 5페이지)

**Day 14 — RC freeze & 문서/체크리스트 고정**

* PRD Stage E 완료 조건을 체크리스트로 고정(Release_Candidate_Freeze)
* 데이터셋 버전(pv26_v1) 체크섬/리포트 보존, 재빌드 재현 확인
* 산출물: 제출 전 “동일 커밋/동일 config로 재현” 되는지 검증 로그

### V2 later (Backlog, decision-ready)

* RLMD를 lane supervision에 포함하기 위한 **lane-boundary-equivalent mapping table** 버전관리 + 검증(현재는 제외가 타당) 
* INT8(또는 QAT)까지 포함한 배포 최적화(정확도 회귀 게이트 포함) 
* camera_id-conditioned BN/어댑터(실제 도메인 갭이 확인되면 적용) 
* obstacle segmentation(V2 범위)와 semantic_id 확장(현재 SPADE 계약이 고정이므로 MVP에서는 금지)
* multi-camera batch inference(2장을 한 번에 넣는 batching)와 CUDA Graph 적용(성능 병목이 GPU kernel launch로 확인될 때)
