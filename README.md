# YOLOPV26

This repository is scoped to 2D perception training and export work for YOLOPV26.

## Package Layout

```text
model/
  preprocess/
    aihub_common.py
    aihub_standardize.py
  viz/
    overlay.py
  loss/
    spec.py
tools/
test/
docs/
```

- `model/` is the project package root.
- `model/preprocess/` holds source parsing and standardization code.
- `model/viz/` holds human QA visualization utilities.
- `model/loss/` holds training-loss specifications.

## Docs

Start from [0_PRD.md](/home/user1/ROS2_Workspace/ros2_ws/src/YOLOpv26/docs/0_PRD.md). The numbered docs under `docs/` are the only active design documents for this repository.

## AIHUB Standardization

The current AIHUB preprocessing deliverable is a hardcoded standardization pipeline for PV26 training preparation. It:

- scans the local AIHUB lane and traffic roots
- writes source-dataset README files back into the original AIHUB directories
- preserves the source dataset intact and materializes converted outputs under `seg_dataset/pv26_aihub_standardized`
- emits real-time stage logs, progress, throughput, and ETA
- normalizes traffic scenes into `7-class OD + traffic_light 4-bit attributes`
- preserves AIHUB lane, stop-line, and crosswalk geometry in scene JSON for later target encoding

### Run

```bash
python3 -m model.preprocess.aihub_standardize --max-samples-per-dataset 64
```

or

```bash
python3 tools/run_aihub_standardize.py --max-samples-per-dataset 64
```

### Outputs

```text
seg_dataset/pv26_aihub_standardized/
  images/<split>/*
  labels_det/<split>/*.txt
  labels_scene/<split>/*.json
  meta/
    class_map_det.yaml
    class_map_scene.yaml
    conversion_report.json
    conversion_report.md
    source_inventory.json
    source_inventory.md
    debug_vis/
      <split>/*.png
      index.json
```

## Loss Spec

The active loss design is described in [5_TARGETS_AND_LOSS.md](/home/user1/ROS2_Workspace/ros2_ws/src/YOLOpv26/docs/5_TARGETS_AND_LOSS.md) and mirrored as code in [model/loss/spec.py](/home/user1/ROS2_Workspace/ros2_ws/src/YOLOpv26/model/loss/spec.py).
