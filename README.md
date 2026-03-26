# YOLOPV26

This repository is scoped to 2D perception training and export work for YOLOPV26.

## Package Layout

```text
model/
  encoding/
    pv26_target_encoder.py
  eval/
    pv26_evaluator.py
  loading/
    pv26_loader.py
    transform.py
  heads/
    pv26_heads.py
  trunk/
    ultralytics_yolo26.py
  preprocess/
    aihub_common.py
    aihub_standardize.py
    bdd100k_standardize.py
  viz/
    overlay.py
  loss/
    spec.py
    runtime.py
  training/
    pv26_trainer.py
tools/
test/
docs/
```

- `model/` is the project package root.
- `model/preprocess/` holds source parsing and standardization code.
- `model/loading/` holds canonical dataset loading and shared online transforms.
- `model/encoding/` holds fixed-shape target encoding from collated samples.
- `model/eval/` holds batch-level evaluation skeletons and summary helpers.
- `model/heads/` holds PV26 custom multitask head modules.
- `model/trunk/` holds pretrained trunk adapters and partial weight-loading helpers.
- `model/viz/` holds human QA visualization utilities.
- `model/loss/` holds training-loss specifications and smoke/runtime loss code.
- `model/training/` holds stage configuration, optimizer wiring, and train-step skeletons.

## Docs

Start from [0_PRD.md](docs/0_PRD.md). The numbered docs under `docs/` are the only active design documents for this repository.

## AIHUB Standardization

The current AIHUB preprocessing deliverable is a hardcoded standardization pipeline for PV26 training preparation. It:

- scans the local AIHUB lane, obstacle, and traffic roots
- writes source-dataset README files back into the original AIHUB directories
- preserves the source dataset intact and materializes converted outputs under `seg_dataset/pv26_aihub_standardized`
- emits real-time stage logs, progress, throughput, and ETA
- supports resume scan on existing standardized outputs and `--force-reprocess` when a clean rebuild is needed
- writes failure manifest and QA summary artifacts for long-running full-dataset conversion
- resolves dataset roots from repo-relative defaults or `PV26_*` environment overrides instead of host-only absolute paths
- normalizes obstacle scenes into detector-only `traffic_cone / obstacle` canonical outputs
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
    failure_manifest.json
    failure_manifest.md
    qa_summary.json
    qa_summary.md
    source_inventory.json
    source_inventory.md
    debug_vis/
      <split>/*.png
      index.json
```

## BDD100K Standardization

The current BDD100K preprocessing deliverable is a hardcoded detection-only standardization pipeline for PV26. It:

- scans `bdd100k_images_100k/100k/<split>` and `bdd100k_labels/100k/<split>`
- writes a source-dataset README back into the BDD100K root
- preserves the source dataset intact and materializes converted outputs under `seg_dataset/pv26_bdd100k_standardized`
- emits real-time stage logs, progress, throughput, and ETA
- supports resume scan on existing standardized outputs and `--force-reprocess` when a clean rebuild is needed
- writes failure manifest and QA summary artifacts for long-running full-dataset conversion
- resolves dataset roots from repo-relative defaults or `PV26_*` environment overrides instead of host-only absolute paths
- collapses BDD categories into the PV26 7-class OD taxonomy
- preserves BDD weather/scene/timeofday metadata and traffic-light color hints in scene JSON while keeping TL supervision disabled for this source

### Run

```bash
python3 -m model.preprocess.bdd100k_standardize --max-samples-per-split 64
```

or

```bash
python3 tools/run_bdd100k_standardize.py --max-samples-per-split 64
```

### Outputs

```text
seg_dataset/pv26_bdd100k_standardized/
  images/<split>/*
  labels_det/<split>/*.txt
  labels_scene/<split>/*.json
  meta/
    class_map_det.yaml
    class_map_scene.yaml
    conversion_report.json
    conversion_report.md
    failure_manifest.json
    failure_manifest.md
    qa_summary.json
    qa_summary.md
    source_inventory.json
    source_inventory.md
    debug_vis/
      <split>/*.png
      index.json
```

## Loss

The active sample contract is described in [4A_SAMPLE_AND_TRANSFORM_CONTRACT.md](docs/4A_SAMPLE_AND_TRANSFORM_CONTRACT.md). The active loss design is described in [5_TARGETS_AND_LOSS.md](docs/5_TARGETS_AND_LOSS.md) and mirrored as code in [model/loss/spec.py](model/loss/spec.py).

Current smoke/runtime loss lives in [model/loss/runtime.py](model/loss/runtime.py). It supports finite multitask loss computation and backward smoke for:

- `det`
- `tl_attr`
- `lane`
- `stop_line`
- `crosswalk`

## YOLO26 Preflight

Current supported smoke environment is `ultralytics 8.4.25 + torch 2.10.0 + torchvision 0.25.0 + numpy 1.26.4`.

```bash
python3 tools/check_env.py --check-yolo-runtime
```

This command performs environment preflight and can also run a real `YOLO("yolo26n.pt")` load when `--check-yolo-runtime` is enabled.

## Preflight

```bash
python3 tools/check_env.py
python3 tools/check_env.py --check-yolo-runtime --strict
```

This command reports the current runtime stack, checks YOLO26 support, and verifies whether `torchvision.ops.nms` is callable. PV26 postprocess now has a pure PyTorch NMS fallback, so missing torchvision NMS is no longer an import-time blocker.

## Tiny Overfit Smoke

```bash
python3 tools/run_pv26_tiny_overfit_smoke.py
```

Edit the config block at the top of `tools/run_pv26_tiny_overfit_smoke.py`, then run the script. It builds a mixed tiny batch from canonical train samples, runs repeated train steps, and prints the loss history as JSON.

## Train

```bash
python3 tools/run_pv26_train.py
```

`tools/run_pv26_train.py` is now a meta-train scenario runner. Edit the scenario YAML at [config/pv26_meta_train.default.yaml](config/pv26_meta_train.default.yaml), or point the top-level `ENTRY_CONFIG.scenario_path` in [tools/run_pv26_train.py](tools/run_pv26_train.py) at another YAML, then run the script.

The runner executes the fixed 3-phase curriculum automatically:

- `stage_1_frozen_trunk_warmup`
- `stage_2_partial_unfreeze`
- `stage_3_end_to_end_finetune`

Each phase uses the YAML policy for:

- `min_epochs`
- `max_epochs`
- `patience`
- `min_improvement_pct`
- per-phase optimizer and runtime overrides

Phase promotion is hybrid:

- the phase cannot advance before `min_epochs`
- after that, it keeps tracking `selection.metric_path`
- if relative improvement stays below `min_improvement_pct` for `patience` validation epochs, the next phase starts
- `max_epochs` is always the hard cap

Phase-internal crash recovery still uses `auto_resume`, but phase-to-phase handoff is now weights-only from the previous phase `best.pt`. That removes the old ambiguity where `stage` and `auto_resume` were mixed together.

Artifacts are written under `runs/pv26_meta_train/<meta_run_name>/`:

- `meta_manifest.json`
- `summary.json`
- `phase_1/`, `phase_2/`, `phase_3/`
- `preview/phase_1/`, `preview/phase_2/`, `preview/phase_3/`

Each phase keeps its own checkpoints, JSONL histories, TensorBoard logs, and `run_manifest.json`. The meta-run summary tracks completed phases, best checkpoints, and promotion reasons.

Validation loaders in this command are sequential eval loaders, not balanced train samplers.

### Preview Bundles

When `preview.enabled=true`, the runner also exports fixed-sample inference previews for each phase:

- `best`: the best checkpoint selected by `selection.metric_path`
- `last`: the phase-end checkpoint

Each preview bundle writes:

- prediction JSON per sample
- overlay PNG per sample
- `index.json`

### TensorBoard

`tools/run_pv26_train.py` now defaults to `tensorboard_mode="curated"`. This keeps the Scalars tab focused on the tags that are useful every day, while `tensorboard_mode="full"` preserves the old verbose debug dump.

Recommended cards for routine monitoring:

- `epoch/train/loss_mean/total` and `epoch/val/loss_mean/total`
- `epoch/train/loss_mean/det`, `epoch/train/loss_mean/lane`, `epoch/train/loss_mean/crosswalk`
- `epoch/scheduler/lr/0` and `epoch/scheduler/lr/1`
- `train_progress/profile_mean/iteration_sec`, `train_progress/profile_mean/load_sec`, `train_progress/profile_mean/forward_sec`
- `train_step/health/skipped_steps` and `train_step/health/gradient_scale`

Use `train_step/*` to inspect noisy batch-level behavior, `train_progress/*` to inspect throughput and ETA, and `epoch/*` to judge whether training quality is improving.
