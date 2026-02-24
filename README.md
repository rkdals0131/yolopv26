# YOLOPv26

## BDD100K 변환 가이드 (현재 구현 스크립트 기준)

현재 저장소는 **BDD100K 전용 Type-A 변환 파이프라인(1차 실행 슬라이스)**을 제공합니다.  
즉, BDD100K 원본 라벨(JSON/마스크)을 학습용 산출물로 변환하고, 검증/품질확인/시각화까지 한 번에 처리할 수 있습니다.

### 지원 기능
- **Detection 변환**: BDD JSON 객체 라벨을 표준 **YOLO txt** 형식으로 변환
- **Drivable Area 변환**: BDD drivable id PNG를 `{0,1,255}` 도메인의 픽셀 마스크로 변환
- **Road Marking 변환**: BDD `lane/*` poly2d를 rasterize하여 RM 마스크 생성
  - `lane_marker`, `road_marker_non_lane`, `stop_line`
- **메타 산출물 생성**: `split_manifest.csv`, `class_map.yaml`, `conversion_report.json`, `checksums.sha256`
- **검증 도구**: manifest/파일존재/마스크도메인/partial-label 규약 검사
- **QC 리포트**: split/has_* 분포, 태그 분포, mask non-empty 비율 집계
- **디버그 시각화**: 마스크 컬러맵 + 원본 오버레이 PNG 생성
- **인터랙티브 실행기**: 변환→검증→QC→디버그(옵션) 순서 자동 실행

### 입력 경로 규칙
`tools/run_bdd100k_normalize_interactive.py` 기준으로, `--bdd-root` 아래에 다음 구조가 필요합니다.

- `bdd100k_images_100k/100k`
- `bdd100k_labels/100k`
- `bdd100k_drivable_maps/labels`

### 변환 방법 1) 단일 CLI로 직접 실행
```bash
python tools/convert_bdd_type_a.py \
  --images-root datasets/BDD100K/bdd100k_images_100k/100k \
  --labels datasets/BDD100K/bdd100k_labels/100k \
  --drivable-root datasets/BDD100K/bdd100k_drivable_maps/labels \
  --out-root datasets/pv26_v1_bdd \
  --splits train,val \
  --include-rain \
  --include-night \
  --allow-unknown-tags
```

### 변환 방법 2) 인터랙티브 파이프라인 실행
```bash
python tools/run_bdd100k_normalize_interactive.py --bdd-root datasets/BDD100K
```
실행 중 질문에 따라 `limit`, `split`, 검증/디버그 실행 여부 등을 선택할 수 있습니다.

### 변환 후 검증
```bash
python tools/validate_pv26_dataset.py --out-root datasets/pv26_v1_bdd
```

### QC 리포트 생성
```bash
python tools/pv26_qc_report.py \
  --dataset-root datasets/pv26_v1_bdd \
  --out-json datasets/pv26_v1_bdd/meta/qc_report.json
```

### 디버그 시각화 생성
```bash
python tools/render_pv26_debug_masks.py \
  --dataset-root datasets/pv26_v1_bdd \
  --split val \
  --channels da,rm_lane_marker,rm_road_marker_non_lane,rm_stop_line \
  --num-samples 20 \
  --out-root /tmp/pv26_mask_vis
```

### 주요 산출물 위치
- 이미지: `images/<split>/*.jpg`
- Detection: `labels_det/<split>/*.txt`
- DA: `labels_seg_da/<split>/*.png`
- RM: `labels_seg_rm_*/<split>/*.png`
- Manifest: `meta/split_manifest.csv`
- 변환 리포트: `meta/conversion_report.json`
- QC 리포트: `meta/qc_report.json`
- 디버그 시각화(선택): `<out-root>/meta/debug_vis` 또는 `--out-root` 지정 경로
