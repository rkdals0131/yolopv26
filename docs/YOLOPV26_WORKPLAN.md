# YOLOPv26 작업 계획 (BDD100K Type-A 기준)

이 문서는 **BDD100K → PV26 Type-A 변환 산출물**을 입력으로, **YOLOPv26 멀티태스크 모델(OD + Drivable + RoadMarking)**을 “정석대로(문서 계약 기반)” 구현/학습하기 위한 **단일 작업 계획서**입니다.

---

## 0) 기준 문서 / 레퍼런스

- 제품/데이터/학습 계약(필수): `docs/PRD.md`
- 데이터 변환/검증 계약(필수): `docs/DATASET_CONVERSION_SPEC.md`
- 현재 구현된 BDD-only 변환 파이프라인(필수): `docs/PV26_TYPEA_IMPLEMENTATION.md`
- 모델 설계 참고(권장): `docs/papers/YOLOPv2 - Better, Faster, Stronger for Panoptic Driving Perception.pdf`
  - 핵심 참고 포인트:
    - 공유 인코더 + **3개 헤드 분리**(det/drivable/lane)
    - **Drivable 헤드는 얕은 feature(pre-FPN)에서 분기 + 업샘플 보강**
    - **Lane 헤드는 깊은 feature(post-FPN)에서 분기 + deconv 사용**

---

## 1) 목표 스코프 (MVP)

### 1.1 입력 (전처리 산출물)
PV26 변환 스크립트가 생성한 아래 구조를 학습 입력으로 사용합니다.

- `images/<split>/*.jpg`
- `labels_det/<split>/*.txt` (YOLO txt)
- `labels_seg_da/<split>/*.png` (uint8, `{0,1,255}`)
- `labels_seg_rm_lane_marker/<split>/*.png` (uint8, `{0,1,255}`)
- `labels_seg_rm_road_marker_non_lane/<split>/*.png` (uint8, `{0,1,255}`)
- `labels_seg_rm_stop_line/<split>/*.png` (uint8, `{0,1,255}`)
- `meta/split_manifest.csv` (학습/검증의 source-of-truth)

### 1.2 출력 (모델 태스크)
`docs/PRD.md` 계약을 그대로 따릅니다.

- **OD (Detection)**: canonical 11 클래스(고정)  
  (`car,bus,truck,motorcycle,bicycle,pedestrian,traffic_cone,barrier,bollard,road_obstacle,sign_pole`)
- **DA (Drivable Area)**: 1채널 binary segmentation
- **RM (RoadMarking)**: 3채널 binary segmentation
  - `lane_marker`
  - `road_marker_non_lane`
  - `stop_line`

> 참고: `semantic_id`(mono8)은 배포/연동용 산출물이며, 학습의 1차 목표는 multi-channel RM + DA입니다.

### 1.3 Out-of-scope (당장 안 함)
- Instance segmentation
- BDD 원본 taxonomy 전체를 그대로 사용하는 “raw class 그대로 학습” 방식  
  (현재 PRD/SPEC/변환기 모두 canonical 정책 기반)

---

## 2) 전처리(변환) 실행 절차 (재현 가능한 1회 run 확보)

### 2.1 인터랙티브 실행(권장)
사용자는 CLI 옵션을 최소화하고 질문에 답해서 실행합니다.

```bash
python tools/run_bdd100k_normalize_interactive.py --bdd-root <BDD100K_ROOT>
```

이 스크립트는 아래 단계를 순서대로 수행합니다.

1. `tools/convert_bdd_type_a.py` (변환)
2. `tools/validate_pv26_dataset.py` (선택)
3. `tools/pv26_qc_report.py` (항상)
4. `tools/render_pv26_debug_masks.py` (선택)

### 2.2 필수 산출물 체크
변환이 끝나면 최소한 아래 두 파일을 확인합니다.

- `meta/conversion_report.json` (분포/스킵/설정)
- `meta/split_manifest.csv` (학습의 기준 테이블)

### 2.3 현재 변환 스냅샷 (datasets/pv26_v1_bdd_full)
아래 내용은 **현재 생성된** `datasets/pv26_v1_bdd_full/meta/*` 기준 요약입니다.

- 변환 시각(UTC): `2026-02-24T06:06:36+00:00` (`meta/conversion_report.json`)
- 도메인 필터: `dry + day`만 포함 (`include_rain=false`, `include_night=false`, `allow_unknown_tags=false`)
- 전체 샘플 수: `34,749`
- split 분포:
  - train: `24,306`
  - val: `3,441`
  - test: `7,002`
- 라벨 가용성(has_*) 요약:
  - `has_det=1`: `34,749` (100%)
  - `has_da=1`: `27,747` (대부분 train/val), `has_da=0`: `7,002` (test)
  - `has_rm_lane_marker=1`: `34,749` (100%)
  - `has_rm_road_marker_non_lane=1`: `34,749` (100%)
  - `has_rm_stop_line=1`: `0` (BDD에서는 stop_line supervision 없음)
  - `has_semantic_id=1`: `0` (현재 빌드에서는 semantic_id 미생성)
- Seg non-empty ratio(감시용 지표):
  - DA: `0.9579` (supervised 중)
  - RM lane_marker: `0.8514`
  - RM road_marker_non_lane: `0.9147`
- 검증 결과: `errors=0`, `warnings=0` (`meta/run_manifest.json` validate stage)

---

## 3) 데이터 로더 설계 (Manifest 중심, 부분 라벨 규칙 준수)

학습/검증 로더는 **폴더를 “추정”하지 않고**, 항상 `meta/split_manifest.csv`를 기준으로 샘플을 로드합니다.

### 3.1 핵심 규칙 (partial-label)
`docs/PRD.md` 계약을 그대로 코드로 옮깁니다.

- Segmentation:
  - `ignore index = 255` 픽셀은 loss 계산에서 제외
  - `has_* = 0`이면 해당 태스크 loss를 0으로 마스킹
- Detection:
  - `det_label_scope=none`이면 det loss는 0
  - `det_label_scope=subset`이면 **미주석 클래스에 대한 negative loss 마스킹**
    - (초기에는 BDD-only slice에서 subset이 드물 수 있으나, 설계는 미리 넣어둠)

### 3.2 기본 입력 크기(정석)
`docs/PRD.md`의 권장 입력을 우선 채택합니다.

- 원본: `960x540`
- 학습 입력(정합): letterbox `960x544` (stride 정합)

---

## 4) 모델 아키텍처 설계 (YOLOPv2 참고 + PRD 준수)

### 4.1 큰 구조
- Shared Backbone: YOLO26 계열
- Neck: PAN/FPN 계열
- Heads:
  1) OD head (multi-scale 권장)
  2) DA head (pre-FPN shallow feature에서 분기 + 업샘플)
  3) RM head (post-FPN fused feature에서 분기 + deconv 포함)

### 4.2 현재 구현(스텁)과의 연결
`pv26/multitask_model.py`는 “smoke test / 형태 고정”을 위한 스텁이며,
아래 순서로 발전시킵니다.

1. (현재) **3-head 분리 + DA는 shallow 분기, RM은 deep 분기(+deconv)** 형태 확보
2. (다음) detection head를 YOLO26 방식으로 교체(멀티스케일/디코딩 포함)
3. (다음) 학습 criterion(손실) + 마스킹 규칙을 별도 모듈로 분리

---

## 5) 학습 Loss 설계 (정석 기본값)

`docs/PRD.md`의 기본 조합을 그대로 채택하고, 필요 시 튜닝합니다.

- OD: YOLO detection loss (cls/obj/box)
- DA: CE/BCE + `ignore=255` 마스킹
- RM: Focal + Dice + `ignore=255` 마스킹

기본 loss weight:
- `w_od=1.0`
- `w_da=1.0`
- `w_rm=2.0`

---

## 6) 평가/리포팅 (QC + 학습 metric)

### 6.1 데이터 QC (변환 직후)
- `tools/pv26_qc_report.py` 결과(`meta/qc_report.json`)로:
  - split별 `has_*` 분포
  - 태그(`weather_tag/time_tag/scene_tag`) 분포
  - mask non-empty 비율
  을 확인합니다.

### 6.2 모델 metric (학습 중)
- OD: mAP (canonical 11 classes)
- DA: mIoU (단, `has_da=1` 샘플 + ignore 제외)
- RM: 채널별 IoU (단, 채널별 `has_rm_*=1` 샘플 + ignore 제외)

---

## 7) 구현 단계(체크리스트)

- [x] **1. 변환 재현성 확보 (지금 컨버팅 중이면 여기부터 체크)**
  - [x] 1.1 변환 1회 run 완료 (out_root 고정)
  - [x] 1.2 `meta/split_manifest.csv` 생성 확인(행 수/컬럼)
  - [x] 1.3 `tools/validate_pv26_dataset.py` 통과(권장)
  - [x] 1.4 `meta/conversion_report.json` + `meta/qc_report.json` 생성/요약 기록

- [x] **2. Manifest 기반 Dataset/Dataloader 구현**
  - [x] 2.1 `meta/split_manifest.csv` 기반 샘플 인덱싱(폴더 추정 금지)
  - [x] 2.2 이미지/OD(txt)/Seg(png) 로딩 + 기본 무결성 체크(shape/domain)
  - [x] 2.3 입력 전처리: letterbox `960x544`
    - [ ] 2.3.1 (선택) 동기 기하 증강(flip/scale/rotate 등)
  - [x] 2.4 partial-label 반영: `ignore=255`, `has_*`, `det_label_scope`

- [x] **3. Criterion(손실) 모듈 구현**
  - 구현 파일: `pv26/criterion.py`
  - 검증 테스트: `tests/test_criterion.py`
  - [x] 3.1 OD: YOLO det loss + `det_label_scope` 마스킹(특히 `subset`)
  - [x] 3.2 DA: CE/BCE + `ignore=255` 제외 + `has_da=0` 마스킹
  - [x] 3.3 RM: Focal+Dice + `ignore=255` 제외 + 채널별 `has_rm_*=0` 마스킹
  - [x] 3.4 기본 weight 적용: `w_od=1.0`, `w_da=1.0`, `w_rm=2.0`

- [x] **4. 훈련 스크립트(최소)**
  - 구현 파일: `tools/train_pv26_smoke.py`
  - [x] 4.1 seed/로그/체크포인트 경로 고정
  - [x] 4.2 train/val split 로딩 + 1 epoch smoke run
  - [x] 4.3 metric 계산
    - [x] OD mAP@0.5 (map50)
    - [x] DA/RM IoU (ignore=255 제외 + has_* 샘플만)

- [ ] **5. 디버그 루프**
  - [x] 5.1 `render_pv26_debug_masks.py`로 샘플 시각 확인(변환 산출물 sanity)
  - [ ] 5.2 문제 발생 시 역추적: converter → validator → QC → loader → criterion 순서로 확인

---

## 8) 리스크 / 주의사항

- BDD road-marking(특히 stop_line)은 희소하거나 스키마 상 “없음”으로 나올 수 있어, `has_rm_stop_line=0` 비중이 높을 수 있습니다.  
  → 학습 시 채널별 positive 비율을 먼저 보고(=QC) loss weight/focal 파라미터를 조정합니다.
- “BDD 원본 taxonomy 전체를 그대로 학습”으로 요구가 바뀌면 PRD/SPEC/변환/학습 전부 재설계가 필요합니다.  
  → 현재는 canonical 정책을 유지합니다.
