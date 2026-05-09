# PV26 Lane-Family 60% Continuation - 2026-05-09

## Objective

Continue the 60% lane-family target from branch `exp/lane-family-60pct` without repeating the earlier threshold-only and heads/upper-trunk probes.

This pass treats completion as actual metric evidence above 60%, not as tool or probe completion.

## Current Baseline

Source checkpoint:

- Run: `runs/pv26_exhaustive_od_lane_train/exhaustive_od_lane_default_20260505_032217`
- Checkpoint: `phase_4/checkpoints/best.pt`

Previously confirmed val128 postprocess best:

| Variant | Lane F1 | Stop-line F1 | Crosswalk F1 | Proxy |
| --- | ---: | ---: | ---: | ---: |
| baseline 0.5/0.5/0.5 | 0.3694 | 0.1739 | 0.4856 | 0.3340 |
| threshold combined 0.8/0.8/0.5 | 0.3955 | 0.1846 | 0.4856 | 0.3503 |

## Oracle And Pixel Diagnostics

GT-rendered seg-first lane oracle:

| Signal | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| geometry | 0.8193 | 0.8390 | 0.8290 |
| full schema | 0.7896 | 0.8086 | 0.7990 |

Interpretation: the GT-to-vectorizer path can exceed 60%. The current ceiling is not primarily a vectorizer upper-bound problem.

Checkpoint dense-map pixel PR on val128:

| Map | Best threshold | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| lane centerline core | 0.9 | 0.3465 | 0.7929 | 0.4822 |
| lane support | 0.9 | 0.7554 | 0.8379 | 0.7945 |
| stop-line mask | 0.8 | 0.7019 | 0.5532 | 0.6188 |
| stop-line center | 0.9 | 0.0489 | 0.1844 | 0.0773 |
| crosswalk mask | 0.5 | 0.8710 | 0.8538 | 0.8623 |
| crosswalk center | 0.1 | 0.1147 | 0.1589 | 0.1332 |

Interpretation:

- Lane support is strong, but centerline core is too broad/noisy for the current vectorizer.
- Stop-line mask is useful, but the center/selector signal is weak.
- Crosswalk mask is already strong; crosswalk is not the first bottleneck.

## Decode Variants

`tools/probe_pv26_lane60_decode_variants.py` evaluates postprocess-only variants from a checkpoint without retraining.

Best val128 result:

| Variant | Lane F1 | Stop-line F1 | Crosswalk F1 | Mean F1 | Proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lane_t090_stop_mask_only` | 0.3946 | 0.2044 | 0.4856 | 0.3615 | 0.3557 |
| `lane_t080` | 0.3955 | 0.1739 | 0.4856 | 0.3517 | 0.3470 |
| baseline | 0.3694 | 0.1739 | 0.4856 | 0.3430 | 0.3340 |

Support-map-as-centerline and centerline/support blend variants were worse. This means the support map cannot simply replace centerline input in the current vectorizer.

## Dense Sharpen Probe

New probe axis: `dense_sharpen_rebalance`.

Settings:

- `freeze_policy=lane_family_heads_only`
- lane/stop/cross loss weights: `1.75 / 2.25 / 1.0`
- lane seg-first weights: centerline BCE/Dice increased to `1.5 / 1.5`, support reduced to `0.15`
- stop-line center target: `heatmap`
- stop-line selector weight reduced to `0.5`
- stop-line geometry/local-x weights increased to `1.5 / 0.5`

Short smoke:

| Epochs | Train batches | Val batches | Objective | Lane F1 | Stop-line F1 | Crosswalk F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 32 | 0.5320 | 0.3654 | 0.1379 | 0.4571 |

Interpretation: the configuration runs and is numerically stable, but the small smoke did not show a strong 60% direction.

Larger val128 probes:

| Probe | Seed | Epochs | Objective | Lane F1 | Stop-line F1 | Crosswalk F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dense_sharpen_rebalance` | source best | 2 | 0.5308 | 0.3654 | 0.4027 | 0.5254 |
| `heads_rebalance` | dense best | 2 | 0.5300 | 0.3643 | 0.4054 | 0.5021 |
| `upper_trunk_rebalance` | dense best | 2 | 0.5288 | 0.3637 | 0.4000 | 0.5062 |
| `dense_sharpen_rebalance` | dense best | 4 | 0.5295 | 0.3644 | 0.4000 | 0.5000 |

The stop-line head can be lifted into the `0.40` F1 range, but lane remains pinned around `0.36` and longer same-axis continuation regresses after epoch 2.

Expanded decode sweep on the best dense-sharpen checkpoint:

| Variant | Lane F1 | Stop-line F1 | Crosswalk F1 | Proxy |
| --- | ---: | ---: | ---: | ---: |
| `lane_t090_stop_mask_only_stop_obj070` | 0.3880 | 0.2388 | 0.5039 | 0.3664 |
| `lane_t090_stop_mask_only_cross_mask030` | 0.3880 | 0.2286 | 0.5100 | 0.3646 |
| `lane_t090_stop_mask_only` | 0.3880 | 0.2286 | 0.5039 | 0.3634 |

Threshold tuning only adds about one proxy point over the previous decode result. It does not expose hidden 60% headroom.

## Decision

Not achieved.

What this continuation falsified:

- Support-map substitution is not a shortcut to 60%.
- Length/bottom-y filtering does not materially raise lane F1 on this checkpoint.
- Stop-line mask-only decode helps, but only from `0.1739` to about `0.2044`.
- Stop-line-focused dense sharpening helps stop-line F1, but does not move lane enough.
- Longer same-axis continuation does not climb toward 60%; it peaks by epoch 2 and then regresses.
- Opening the upper trunk at low LR is slower and slightly worse than head-only.

Next useful axis:

1. Preserve `lane_t090_stop_mask_only_stop_obj070` as a postprocess candidate, but do not confuse it with a solution.
2. Stop using same-axis longer continuation as the primary plan; the probe evidence is already negative.
3. The next architectural target is a different lane centerline objective/decoder, not more threshold tuning or trunk unfreezing.

## Commands

Decode variants:

```bash
.venv/bin/python tools/probe_pv26_lane60_decode_variants.py \
  --checkpoint /home/kai/yolopv26/runs/pv26_exhaustive_od_lane_train/exhaustive_od_lane_default_20260505_032217/phase_4/checkpoints/best.pt \
  --max-val-batches 128 \
  --device cuda:0
```

Dense sharpen smoke:

```bash
.venv/bin/python tools/run_pv26_lane60_probe.py \
  --source-run /home/kai/yolopv26/runs/pv26_exhaustive_od_lane_train/exhaustive_od_lane_default_20260505_032217 \
  --experiment dense_sharpen_rebalance \
  --epochs 1 \
  --train-batches 64 \
  --val-batches 32 \
  --batch-size 4 \
  --device cuda:0 \
  --run-root runs/pv26_exhaustive_od_lane_train
```
