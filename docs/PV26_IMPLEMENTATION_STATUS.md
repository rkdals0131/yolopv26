# PV26 Type-A — Dataset Adapters Status

This repo contains runnable **PV26 Type‑A** dataset converters (adapters) that export the canonical layout + manifest.

Implemented adapters:
- **BDD100K**: OD + DA + RM(mask + subclass) (implemented)
- **ETRI (Mono+Multi)**: polygon JSON → DA + RM(mask + subclass) (OD 없음)
- **RLMD (1080p + AC labeled)**: palette RGB mask → RM(mask + subclass) (OD/DA 없음)
- **Waymo / WOD (Perception v2 parquet, minimal-first)**: parquet → panoptic-derived OD(full) + DA + RM(mask) (stop line / lane subclass 없음)

Contracts implemented:
- `docs/PV26_PRD.md` (partial-label policy, classmap-v3)
- `docs/PV26_DATASET_CONVERSION_SPEC.md` (directory layout, mask domains, manifest columns)

## 0) Multi-head architecture stub (for implementation bootstrap)

`pv26/model/multitask_stub.py` + `pv26/model/multitask_yolo26.py` provide a runnable **shared-trunk + 4-head** structure:
- Detection head (dense logits): `det` tensor
- Drivable head: `da` logits, 1 channel
- Road-marking head: `rm` logits, 3 channels
- Lane-subclass head: `rm_lane_subclass` logits, 5 channels (`bg + 4 subclasses`)

This is a shape/interface bootstrap for PV26 integration and tests, not a final YOLO26 production graph yet.

## 1) Converter: `tools/data_analysis/bdd/convert_bdd_type_a.py` (BDD100K)

### Expected inputs
- `--images-root`: directory containing BDD images (`.jpg/.jpeg/.png`)
- `--labels`: (optional) either:
  - a directory of **per-image JSON** label files, or
  - a single JSON containing a list of per-image records  
  If omitted, `has_det=0` and `det_label_scope=none` for all samples.
- `--drivable-root`: (optional) directory containing BDD **drivable id masks** (`.png`) with the same stem as the image.
  If omitted or missing per-image mask, `has_da=0` and the exported DA mask is all-255.

### Example
```bash
python tools/data_analysis/bdd/convert_bdd_type_a.py \
  --images-root /path/to/bdd/images \
  --labels /path/to/bdd/per_image_json \
  --drivable-root /path/to/bdd/drivable_id_masks \
  --out-root /tmp/pv26_v1_bdd \
  --seed 0
```

Notes:
- Default domain filter follows MVP policy from the spec: only `weather_tag=dry` and `time_tag=day`.
  - To include rainy samples: add `--include-rain`
  - To include night samples: add `--include-night`
- To include unknown tags: add `--allow-unknown-tags`
- `--splits train,val` can be used when only train/val output is desired

## 2) Validator: `tools/data_analysis/bdd/validate_pv26_dataset.py`

Validates:
- `meta/split_manifest.csv` schema and referenced file existence
- image/mask width/height consistency
- mask value domains (`{0,1,255}` for DA/RM; no `255` allowed in semantic_id)
- partial-label policy (`has_*=0` => corresponding mask must be all-255)
- `det_label_scope=none` => detection txt must be empty

Example:
```bash
python tools/data_analysis/bdd/validate_pv26_dataset.py --out-root /tmp/pv26_v1_bdd
```

Exit codes:
- `0`: valid
- `2`: validation failures

## 3) QC Report: `tools/data_analysis/bdd/pv26_qc_report.py`

Summarizes split distribution and label availability (`has_*`) from `meta/split_manifest.csv`.

Example:
```bash
python tools/data_analysis/bdd/pv26_qc_report.py \
  --dataset-root /tmp/pv26_v1_bdd \
  --out-json /tmp/pv26_v1_bdd/meta/qc_report.json
```

## 4) Debug visualization: `tools/debug/render_pv26_debug_masks.py`

Example:
```bash
python tools/debug/render_pv26_debug_masks.py \
  --dataset-root /tmp/pv26_v1_bdd \
  --split val \
  --channels da,rm_lane_marker,rm_road_marker_non_lane,rm_stop_line \
  --num-samples 20 \
  --out-root /tmp/pv26_mask_vis
```

## 5) Tests

Run:
```bash
python -m unittest -v
```

## 6) Additional Converters (ETRI/RLMD/WOD)

### 6.1 ETRI: `tools/data_analysis/etri/convert_etri_type_a.py`
- Inputs: `datasets/ETRI/MonoCameraSemanticSegmentation` + `datasets/ETRI/Multi Camera Semantic Segmentation`
- Outputs:
  - `has_det=0`, empty detection txt
  - DA from `road`
  - RM lane subclass from `whsol/whdot/yesol/yedot` (others lane-like are `255(ignore)` in subclass)

### 6.2 RLMD: `tools/data_analysis/rlmd/convert_rlmd_type_a.py`
- Inputs: `datasets/RLMD/RLMD_1080p` + `datasets/RLMD/RLMD-AC` (labels가 존재하는 split만)
- Outputs:
  - `has_det=0`, `has_da=0` (DA는 all-255)
  - RM lane marker + non-lane + stop line
  - lane subclass는 white/yellow solid/dashed만 매핑, 나머지 lane-marker 픽셀은 `255(ignore)`

### 6.3 Waymo/WOD: `tools/data_analysis/wod/convert_wod_type_a.py`
- Inputs: `datasets/WaymoOpenDataset/wod_pv2_minimal_1ctx/training` (parquet)
- Outputs:
  - Detection은 panoptic instance 기반 coarse 7-class로 추출 → `det_label_scope=full`
  - Segmentation은 `ROAD/LANE_MARKER/ROAD_MARKER`만 사용
  - `rm_stop_line` + `rm_lane_subclass`는 제공 불가 → all-255, `has_*=0`
