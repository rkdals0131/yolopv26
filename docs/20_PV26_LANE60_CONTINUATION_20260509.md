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

## Core Centerline Target Probe

New probe axis: `lane_segfirst_centerline_target_mode`.

The earlier lane loss trained centerline logits against the soft centerline map. The dense-map PR showed the support map was strong but the core centerline was too broad/noisy for vectorization, so this pass exposed the centerline target mode:

- `soft`: previous behavior, target is `lane_seg_centerline_soft`
- `core`: target is `lane_seg_centerline_core`
- `hybrid`: target is `max(core, 0.5 * soft)`

The probe also carries the active phase loss weights into train/val summaries. Before this fix, the actual criterion used the probe phase weights, but summary `losses.weighted.*.weight` could be recomputed from static stage defaults. That made loss analysis misleading for derived probes; future summaries now report the active criterion weights.

Val128 continuation probes:

| Probe | Seed | Epochs | Best epoch | Objective | Lane F1 | Stop-line F1 | Crosswalk F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `core_centerline_rebalance` | source best | 2 | 2 | 0.5512 | 0.4121 | 0.4054 | 0.4269 |
| `core_centerline_rebalance` | core best | 4 | 2 | 0.5530 | 0.4253 | 0.3947 | 0.4170 |
| `hybrid_centerline_rebalance` | source best | 2 | 2 | 0.5458 | 0.4056 | 0.4054 | 0.4269 |
| `core_cross_retain` | source best | 2 | 2 | 0.5512 | 0.4121 | 0.4054 | 0.4269 |
| `core_centerline_low_lr` | source best | 4 | 3 | 0.5489 | 0.3960 | 0.4255 | 0.4573 |
| `core_centerline_posw8` | source best | 4 | 2 | 0.5428 | 0.3889 | 0.4000 | 0.4348 |
| `core_centerline_refine` | source best | 4 | 2 | 0.5591 | 0.4397 | 0.4027 | 0.4348 |
| `core_centerline_low_lr` | refine best | 4 | 2 | 0.5550 | 0.4410 | 0.3871 | 0.4141 |
| `core_centerline_refine_tangent` | source best | 4 | 2 | 0.5585 | 0.4375 | 0.4027 | 0.4348 |

Epoch trace for the best core continuation:

| Epoch | Objective | Lane F1 | Stop-line F1 | Crosswalk F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5204 | 0.3938 | 0.1818 | 0.4959 |
| 2 | 0.5530 | 0.4253 | 0.3947 | 0.4170 |
| 3 | 0.5483 | 0.4065 | 0.3946 | 0.4579 |
| 4 | 0.5356 | 0.3993 | 0.2837 | 0.4781 |

Interpretation:

- Core target is a real improvement over dense-sharpen, lifting best objective from `0.5308` to `0.5530`.
- The gain comes mainly from lane F1 moving from the mid-`0.36` range to `0.42`.
- Hybrid target retains too much soft target and is worse than core.
- Raising crosswalk task weight does not preserve crosswalk at the objective peak.
- Lowering head LR from `1e-4` to `5e-5` preserves stop-line/crosswalk better, but loses too much lane score and does not beat core.
- Lowering the centerline BCE positive-weight cap from `32` to `8` also loses lane score; it does not fix the false-positive problem.
- Adding a small gated centerline-only refinement branch is positive: best objective rises from `0.5530` to `0.5591`, with lane F1 rising to `0.4397`.
- Continuing the refine-best checkpoint at `5e-5` does not preserve enough objective headroom. It reaches lane F1 `0.4410`, but stop-line/crosswalk balance is weaker and objective stays at `0.5550`.
- Sharing the refined feature with tangent prediction is near-tie but negative: `0.5585` versus the prior `0.5591`. Keep tangent on the base lane feature unless a later probe changes the vectorizer contract.
- The objective still peaks around epoch 2, then oscillates/regresses; another same-axis longer run is not justified as the primary 60% path.

Dense-map PR on the best core checkpoint:

| Map | Best threshold | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| lane centerline core | 0.9 | 0.5278 | 0.6148 | 0.5680 |
| lane support | 0.9 | 0.7528 | 0.8411 | 0.7945 |
| stop-line mask | 0.8 | 0.6973 | 0.5439 | 0.6111 |
| stop-line center | 0.4 | 0.0855 | 0.3112 | 0.1341 |
| crosswalk mask | 0.8 | 0.8774 | 0.8463 | 0.8616 |
| crosswalk center | 0.1 | 0.1075 | 0.1710 | 0.1320 |

Interpretation: core-target training improved lane centerline-core pixel F1 from the source checkpoint's `0.4822` to `0.5680`, but support remains much stronger at `0.7945`. The lane bottleneck is still mostly centerline-map quality before vectorization, not merely a postprocess threshold issue.

Dense-map PR on the best gated-refine checkpoint:

| Map | Best threshold | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| lane centerline core | 0.9 | 0.5218 | 0.6295 | 0.5706 |
| lane support | 0.9 | 0.7588 | 0.8384 | 0.7966 |
| stop-line mask | 0.8 | 0.6992 | 0.5443 | 0.6121 |
| crosswalk mask | 0.8 | 0.8744 | 0.8504 | 0.8622 |

Interpretation: refine does not radically change pixel PR, but it does raise the vectorized lane metric. The next promising axis should keep the gated refinement branch and search around schedule/retention from its best epoch, not return to the plain head.

Refine-tangent follow-up: the vectorizer consumes `lane_seg_tangent_axis` together with the centerline map. Sharing the refined feature with tangent prediction did not beat centerline-only refinement, so that branch was restored to centerline-only behavior after recording the result.

Decode sweep on the best core checkpoint:

| Variant | Lane F1 | Stop-line F1 | Crosswalk F1 | Proxy |
| --- | ---: | ---: | ---: | ---: |
| `stop_mask_only` | 0.4041 | 0.2302 | 0.4941 | 0.3699 |
| baseline | 0.4041 | 0.1806 | 0.4941 | 0.3550 |

Postprocess variants do not expose a hidden 60% path on the improved checkpoint.

## Decision

Not achieved.

What this continuation falsified:

- Support-map substitution is not a shortcut to 60%.
- Length/bottom-y filtering does not materially raise lane F1 on this checkpoint.
- Stop-line mask-only decode helps, but only from `0.1739` to about `0.2044`.
- Stop-line-focused dense sharpening helps stop-line F1, but does not move lane enough.
- Longer same-axis continuation does not climb toward 60%; it peaks by epoch 2 and then regresses.
- Opening the upper trunk at low LR is slower and slightly worse than head-only.
- Core centerline target is the first real architectural improvement in this pass, but it only reaches `0.5530`.
- Hybrid target and crosswalk reweighting do not clear the ceiling.
- Lower-LR continuation is not the crosswalk-retention solution; its best objective is `0.5489`.
- Positive-weight cap `8` is not the precision solution; its best objective is `0.5428`.
- Gated centerline refinement is the current best architecture-side improvement, but still only reaches `0.5591`.
- Decode-only changes remain too small to be the main path.

Next useful axis:

1. Preserve `lane_t090_stop_mask_only_stop_obj070` as a postprocess candidate, but do not confuse it with a solution.
2. Stop using same-axis longer continuation as the primary plan; the probe evidence is already negative.
3. Keep the core centerline target plus gated centerline refinement as the current best lane axis.
4. The next target is crossing the remaining gap from `0.5591` to `0.60` by preserving the refine epoch-2 lane gain while avoiding the epoch-3/4 stop-line/crosswalk oscillation.

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

Core centerline probe:

```bash
.venv/bin/python tools/run_pv26_lane60_probe.py \
  --source-run /home/kai/yolopv26/runs/pv26_exhaustive_od_lane_train/exhaustive_od_lane_default_20260505_032217 \
  --experiment core_centerline_rebalance \
  --epochs 2 \
  --train-batches 512 \
  --val-batches 128 \
  --batch-size 4 \
  --device cuda:0 \
  --run-root runs/pv26_exhaustive_od_lane_train
```
