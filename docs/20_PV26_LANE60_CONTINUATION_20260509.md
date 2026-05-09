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

## Decision

Not achieved.

What this continuation falsified:

- Support-map substitution is not a shortcut to 60%.
- Length/bottom-y filtering does not materially raise lane F1 on this checkpoint.
- Stop-line mask-only decode helps, but only from `0.1739` to about `0.2044`.
- A tiny dense-sharpen training smoke is stable but not immediately better.

Next useful axis:

1. Preserve `lane_t090_stop_mask_only` as a postprocess candidate, but do not confuse it with a solution.
2. If continuing training, run `dense_sharpen_rebalance` at a real probe size before discarding it.
3. If that fails, the next architectural target is a different lane centerline objective/decoder, not more threshold tuning.

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
