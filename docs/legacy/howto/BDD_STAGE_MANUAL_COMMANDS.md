> **Archived (2026-03-05)** — kept for history; may be outdated.  
> Canonical docs: `docs/PV26_PRD.md`, `docs/PV26_DATASET_CONVERSION_SPEC.md`, `docs/PV26_DATASET_SOURCES_AND_MAPPING.md`, `docs/PV26_IMPLEMENTATION_STATUS.md`.

# BDD Stage Manual Commands (No Variables)

아래 명령어는 변수 없이 바로 실행 가능한 형태입니다.

## 2) Validate

```bash
cd /home/user1/Python_Workspace/YOLOPv26
./.venv/bin/python tools/data_analysis/bdd/validate_pv26_dataset.py --out-root /home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full
```

## 3) QC Report

```bash
cd /home/user1/Python_Workspace/YOLOPv26
./.venv/bin/python tools/data_analysis/bdd/pv26_qc_report.py --dataset-root /home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full --out-json /home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full/meta/qc_report.json
```

## 4) Debug Visualization

```bash
cd /home/user1/Python_Workspace/YOLOPv26
./.venv/bin/python tools/debug/render_pv26_debug_masks.py --dataset-root /home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full --split val --channels da,rm_lane_marker --num-samples 50 --out-root /home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full/meta/debug_vis --seed 0
```

## (참고) 1) Convert Full Re-run

이미 변환이 완료된 상태라면 보통 재실행하지 않습니다. 재실행이 필요할 때만 사용하세요.

```bash
cd /home/user1/Python_Workspace/YOLOPv26
./.venv/bin/python tools/data_analysis/bdd/convert_bdd_type_a.py --images-root /home/user1/Python_Workspace/YOLOPv26/datasets/BDD100K/bdd100k_images_100k/100k --labels /home/user1/Python_Workspace/YOLOPv26/datasets/BDD100K/bdd100k_labels/100k --drivable-root /home/user1/Python_Workspace/YOLOPv26/datasets/BDD100K/bdd100k_drivable_maps/labels --out-root /home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full --seed 0 --min-box-area-px 0 --limit 0 --splits train,val,test --include-rain --include-night --allow-unknown-tags
```

## 순차 실행 (2 -> 3 -> 4)

```bash
cd /home/user1/Python_Workspace/YOLOPv26
set -e
./.venv/bin/python tools/data_analysis/bdd/validate_pv26_dataset.py --out-root /home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full
./.venv/bin/python tools/data_analysis/bdd/pv26_qc_report.py --dataset-root /home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full --out-json /home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full/meta/qc_report.json
./.venv/bin/python tools/debug/render_pv26_debug_masks.py --dataset-root /home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full --split val --channels da,rm_lane_marker --num-samples 50 --out-root /home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full/meta/debug_vis --seed 0
```
