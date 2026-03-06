# Commit History

Generated on `2026-03-07` from branch `e2eOD`.

## Recent Commits

| Date | Short SHA | Subject |
|---|---|---|
| 2026-03-07 | `e8e287b` | tools: add PV26 dataset merge utility |
| 2026-03-07 | `2d5c49f` | train: clean up post-hook removal paths |
| 2026-03-07 | `924d74e` | train: refactor modal wrapper defaults |
| 2026-03-07 | `8743ba4` | train: align editable defaults block formatting |
| 2026-03-07 | `c906ff3` | train: add editable local default block |
| 2026-03-07 | `6884550` | train: adopt final PV26 perf defaults |
| 2026-03-07 | `5948e0d` | criterion: decouple ultralytics detection loss |
| 2026-03-07 | `66ae3bd` | model: remove hook-based det feature capture |
| 2026-03-07 | `2887148` | model: add detection backend adapter shell |
| 2026-03-07 | `d3bdd19` | train: add half-res segmentation path |
| 2026-03-07 | `9e5ab34` | model: share road-marking decoder trunk |
| 2026-03-07 | `b322371` | criterion: add optional compiled seg loss block |
| 2026-03-07 | `72e22ce` | criterion: restore pin-memory support for prepared batches |
| 2026-03-07 | `d671b27` | criterion: introduce typed prepared batch hot path |
| 2026-03-07 | `019dace` | criterion: harden merged OD path and keep lane loss variants |
| 2026-03-07 | `e320588` | docs: refine PV26 rollout plan for execution |
| 2026-03-07 | `7222d09` | pv26: skip redundant final interpolate in seg heads |
| 2026-03-06 | `cdc65a2` | multi dataset processing |
| 2026-03-05 | `f12a4f3` | multi dataset processing |
| 2026-03-05 | `431959a` | debug visualizer |
| 2026-03-05 | `0b33989` | docs refactor |
| 2026-03-05 | `0dd82fe` | inference output format compatibility |
| 2026-03-05 | `060b593` | Merge branch 'pv26/train-pipeline-improvements-20260305' into e2eOD |
| 2026-03-05 | `421afca` | train: add present-only lane-subclass mIoU metric |
| 2026-03-05 | `8442f15` | train: split optimizer param groups (trunk/head, no_decay) |
| 2026-03-05 | `99a5029` | pv26: reject det_label_scope=subset for ultralytics OD loss |
| 2026-03-05 | `833e559` | train: save best_total and best_det checkpoints separately |
| 2026-03-05 | `ea40e70` | pv26: supervise lane-subclass CE on positive pixels only |
| 2026-03-05 | `5df9342` | pv26: avoid BN pollution in YOLO26 init dry forward |
| 2026-03-05 | `765fe33` | data preprocessing in parallel, train hyperparams tweak |
| 2026-03-04 | `7a0de16` | feat(pv26): expand type-A dataset pipeline, training tooling, and docs |
| 2026-03-04 | `ec0c617` | modal h100: set tuned loader/resource baseline |
| 2026-03-04 | `5133528` | modal train: add GPU/CUDA preflight and configurable torch specs |
| 2026-03-04 | `fa73c21` | Tune A100 baseline defaults and validation timing controls |
| 2026-03-04 | `0ca5607` | misc parameter tweak |
| 2026-03-03 | `79a07fb` | Optimize PV26 training/data pipeline hot paths |
| 2026-03-03 | `ab47e0a` | Remove per-step sync hotspots and refine compile/scheduler defaults |
| 2026-03-03 | `58f35a5` | Add compile and optimizer/scheduler controls to PV26 training |
| 2026-03-03 | `f485bc0` | Improve step progress labels and set default profile cadence |
| 2026-03-03 | `603c026` | Reduce loader pressure with uint8 images and cleaner wait timing |

## Regenerate

```bash
git log --date=short --pretty=format:'%H%x09%h%x09%ad%x09%s' -40
```
