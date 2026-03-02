# PV26 Type-A (BDD-only) — First Executable Slice

This repo now contains an initial, runnable implementation of **PV26 Type‑A** dataset conversion for **BDD100K** only.

Scope of this slice (intentionally minimal):
- Export PV26 canonical directory layout (`datasets/pv26_v1/` by default)
- Export detection labels (YOLO txt) when BDD per-image JSON is provided
- Export drivable-area mask (`labels_seg_da`) when BDD drivable *id masks* are provided
- Export road-marking masks from BDD `lane/*` poly2d labels:
  - `labels_seg_rm_lane_marker`: rasterized lane-marker classes (`lane/single|double ...`)
  - `labels_seg_rm_road_marker_non_lane`: rasterized non-lane classes (`lane/crosswalk`, `lane/road curb`)
  - `labels_seg_rm_stop_line`: default `255(ignore)` unless explicit stop-line class appears
- Export manifest + conversion report + class_map

Contracts implemented:
- `docs/PRD.md` (partial-label policy, classmap-v2)
- `docs/DATASET_CONVERSION_SPEC.md` (directory layout, mask domains, manifest columns)

## 0) Multi-head architecture stub (for implementation bootstrap)

`pv26/multitask_model.py` provides a runnable **shared-trunk + 3-head** structure:
- Detection head (dense logits): `det` tensor
- Drivable head: `da` logits, 1 channel
- Road-marking head: `rm` logits, 3 channels

This is a shape/interface bootstrap for PV26 integration and tests, not a final YOLO26 production graph yet.

## 1) Converter: `tools/data_analysis/bdd/convert_bdd_type_a.py`

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

## 3) Tests

Run:
```bash
python -m unittest -v
```
