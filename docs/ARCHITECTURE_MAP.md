# PV26 Architecture Map

## Canonical Package Boundaries

- `pv26/dataset/`: dataset contracts, manifests, masks, source adapters, split policy, loading.
- `pv26/model/`: model contracts, backends, heads, multi-task assemblies.
- `pv26/loss/`: criterion and detection/segmentation loss adapters.
- `pv26/training/`: train/val runtime orchestration and factories.
- `pv26/eval/`: detection/segmentation evaluation helpers.
- `pv26/io/`: filesystem/hash/json/time helpers.

## Dependency Direction

- `dataset -> io`
- `model -> (independent as much as possible)`
- `loss -> model/contracts`
- `eval -> model/contracts + dataset/labels`
- `training -> dataset + model + loss + eval`
- `tools/* -> canonical package APIs only`

## Edit Routing

- Class map / manifest / split policy: `pv26/dataset/*`
- YOLO26 backend adapter: `pv26/model/backends/ultralytics_yolo26.py`
- Segmentation loss behavior: `pv26/loss/*`
- Train runtime / optimizer / checkpoint: `pv26/training/*`
- Modal GPU profile defaults: `tools/train/modal/profiles.py`

## Entrypoints

- Local training CLI: `tools/train/train_pv26.py` (thin entrypoint)
- Training runtime: `pv26/training/runner.py`
- Modal launcher: `tools/train/modal/launcher.py`
- Modal profile shims:
  - `tools/train/modal_train_pv26.py`
  - `tools/train/modal_a100_train_pv26.py`
  - `tools/train/modal_h100_train_pv26.py`

