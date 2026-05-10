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
| `core_centerline_refine_open_gate` | source best | 4 | 2 | 0.5588 | 0.4384 | 0.4027 | 0.4297 |
| `core_centerline_refine_cross_retain` | source best | 4 | 2 | 0.5577 | 0.4361 | 0.4000 | 0.4348 |
| `core_centerline_refine_stop_retain` | source best | 3* | 2 | 0.5583 | 0.4383 | 0.3974 | 0.4331 |
| `core_centerline_refine_dice_focus` | source best | 2* | 2 | 0.5523 | 0.4139 | 0.4027 | 0.4348 |
| `core_centerline_refine` | merged task heads | 3* | 2 | 0.5595 | 0.4391 | 0.3893 | 0.4202 |
| `core_centerline_refine_stop_retain` | merged task heads | 2* | 2 | 0.5595 | 0.4390 | 0.3893 | 0.4219 |
| support-conditioned centerline residual | source best | 2* | 2 | 0.5582 | 0.4371 | 0.4027 | 0.4348 |

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
- Opening the refinement gate from `-4` to `-2` is also near-tie but negative: `0.5588` versus `0.5591`. The gain is not limited by a too-closed initial residual gate.
- Raising crosswalk phase weight on top of refinement is negative: crosswalk does not rise at the objective peak, while lane and stop-line slip slightly.
- Raising stop-line phase weight on top of refinement is also near-tie but negative: `0.5583` versus `0.5591`. It does not preserve enough stop-line score to justify the lane/crosswalk trade.
- Shifting centerline loss toward Dice overlap is negative: objective drops to `0.5523`, mainly from lane F1 falling to `0.4139`. The current centerline issue is not fixed by simply making the dense loss more Dice-heavy.
- Seeding from the merged task-head checkpoint is a tiny positive but not a 60% break: objective rises from `0.5591` to `0.5595`, while stop-line/crosswalk balance still limits the result.
- Combining merged-head seeding with stop-line retention only ties the same level (`0.5595`) and does not open new headroom.
- Adding a support-conditioned centerline residual branch was negative (`0.5582`) and was reverted after the probe. The support map remains diagnostically useful, but feeding it directly into the centerline residual did not improve vectorized lane selection.
- Opening the upper trunk at a very low LR from the merged stop-retain checkpoint was also negative: lane F1 reached `0.4426`, but the overall objective stayed at `0.5580` because stop-line/crosswalk balance did not improve.
- Stop/cross-only adaptation did not beat the current best (`0.5584`), but its epoch-1 crosswalk head provided useful merge material.
- Merging the current best lane/stop heads with that crosswalk-adapted head and then running cross-retain adaptation produced the current best: objective `0.5609` at epoch 2.
- The objective still peaks around epoch 2, then oscillates/regresses; another same-axis longer run is not justified as the primary 60% path.

`*` These runs were stopped after the result was clearly below the current best, to return GPU time to the next probe.

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

Open-gate follow-up: the best centerline-refine checkpoint kept `centerline_refine_gate_logit` near `-4` (`sigmoid ~= 0.018`), so the positive signal came from a very small residual branch. Initializing the gate at `-2` (`sigmoid ~= 0.119`) did not beat the original gated refinement, so the default gate was restored to `-4`.

Task-head merge probe: a helper was added to merge `lane_head`, `stop_line_head`, and `crosswalk_head` weights from the task-best checkpoints into one checkpoint. On the refine run this did not produce an immediate threshold-sweep win, and the sweep metric path is not identical to training selection metrics, so merged task heads are not promoted as a 60% candidate without a dedicated selection-metric evaluator.

Merged-head adaptation result: using `merged_task_heads.pt` as the seed and running the normal `core_centerline_refine` probe produced a tiny new best at epoch 2: objective `0.5595`, lane/stop/cross F1 `0.4391 / 0.3893 / 0.4202`. This is the current best numeric probe, but the gain is only `+0.0004` over the previous refine checkpoint and remains far below 60%.

Merged-head stop-retain result: repeating the stop-line retention idea from the merged-head seed reached objective `0.5595`, lane/stop/cross F1 `0.4390 / 0.3893 / 0.4219`. This is effectively a tie with merged-head adaptation, not a new direction.

Support-conditioned residual result: a temporary architecture variant added a small gated residual branch where the centerline logit could see the support probability. It reached only `0.5582` at epoch 2 and was reverted. This falsifies the simple version of "use strong support map to refine centerline"; a later attempt would need a different contract, not just direct support concatenation.

Upper-trunk continuation result: from the merged stop-retain checkpoint, opening the upper trunk with `trunk_lr=2e-6` and `head_lr=1e-4` reached objective `0.5580` at epoch 2. Lane F1 rose to `0.4426`, but stop-line/crosswalk F1 were only `0.3867 / 0.4093`. This says the current ceiling is not simply "frozen trunk prevents lane refinement"; the extra trunk freedom did not raise the multi-task objective.

Stop/cross-only adaptation result: setting lane loss to zero while training stop-line/crosswalk heads reached only `0.5584` at epoch 2, below the previous best. The useful part was task specialization: epoch 1 raised crosswalk F1 to `0.5020`, while epoch 2 kept lane/stop near the old operating point but did not improve the joint objective.

Merged cross-adapt head result: merging lane and stop-line heads from the current merged stop-retain best with the crosswalk head from the stop/cross-only run's `best_crosswalk.pt`, then adapting with `core_centerline_refine_cross_retain`, produced a new best objective `0.5609` at epoch 2. Metrics at that point were lane/stop/cross F1 `0.4435 / 0.3946 / 0.4190`, with component scores `0.6228 / 0.5452 / 0.4268`. This is a real improvement over `0.5595`, but still far below the 60% target.

Second merge follow-up: merging the same epoch-2 lane/stop heads with the epoch-4 crosswalk head from the merged cross-adapt run did not compound the gain. It reached only objective `0.5572` at epoch 2, with lane/stop/cross F1 `0.4407 / 0.3947 / 0.4186`. Repeating task-head merge after crosswalk drift is therefore negative evidence, not a hidden 60% path.

Balanced-retain follow-up: starting from the new `0.5609` best and raising stop-line phase weight to `2.25` while keeping crosswalk at `1.75` produced objective `0.5569` at epoch 2 and `0.5550` at epoch 3. The epoch-2 lane/stop/cross F1 were `0.4405 / 0.3947 / 0.4138`; epoch 3 traded a little crosswalk back to `0.4548` but dropped lane/stop to `0.4302 / 0.3803`. This confirms that the current best is not improved by simply rebalancing stop/cross weights after the merged cross-adapt seed.

Temporary crosswalk-isolator result: a local architecture probe added identity-biased gated residual isolators on P2/P3/P4 before the crosswalk head, matching the stop-line isolation pattern but scoped to crosswalk. From the `0.5609` seed it reached only objective `0.5574` at epoch 2, with lane/stop/cross F1 `0.4430 / 0.3922 / 0.4186`. The lane score was preserved, but crosswalk did not improve and stop-line slipped, so the code was reverted after recording the run.

Refine-cross retention follow-up: the earlier `core_cross_retain` result was measured before the gated centerline refinement branch existed. Repeating that idea on the now-best centerline-only refinement path was still negative, so crosswalk phase weight alone is not the missing 60% lever.

Refine-stop retention follow-up: `core_centerline_low_lr` showed stop-line could rise above the refine run, but at the cost of lane score. The next single-axis probe keeps the normal LR and raises only stop-line phase weight from `1.75` to `2.25` as `core_centerline_refine_stop_retain`.

Refine-stop retention result: this near-tied but did not beat the current best. Epoch 2 reached objective `0.5583` with lane/stop/cross F1 `0.4383 / 0.3974 / 0.4331`, below `core_centerline_refine` at `0.5591`.

Refine-Dice focus result: this was clearly worse. Epoch 2 reached objective `0.5523` with lane/stop/cross F1 `0.4139 / 0.4027 / 0.4348`; increasing centerline Dice weight traded away lane quality.

Support-gated centerline decode probe: `tools/probe_pv26_lane60_support_gate.py` evaluates postprocess-only variants that replace or limit centerline probabilities with support probabilities. On the best refine checkpoint, the best relative variant was `centerline_support_floor070`, but it only moved the probe objective from `0.52806` to `0.52869` on that decode path. This does not expose a hidden 60% path, and the decode-probe metric path should not be confused with the training selection metric.

Postprocess threshold sweep: `tools/probe_pv26_lane60_postprocess_thresholds.py` evaluates lane object, stop-line object/mask, and crosswalk object/mask threshold variants on one cached forward pass. On the current merged stop-retain best checkpoint, the proxy baseline was `0.52994` and the best variant was `stop_mask_0.30__cross_mask_0.40` at `0.53328`. This is useful for calibration, but the gain is too small to explain the missing 60% path.

Exact-config threshold sweep on the current `0.5609` best still does not reproduce the training selection metric because the standalone evaluator path samples validation differently. Its proxy baseline was `0.52915` and the best variant was `stop_mask_0.20` at `0.53380`. Treat this only as decode calibration, not as a replacement for phase selection.

Crosswalk min-rect decode result: replacing the PCA-aligned component box with a convex-hull minimum-area rectangle moved the exact-config proxy baseline from `0.52915` to `0.52970` and the best threshold variant from `0.53380` to `0.53435`. This is too small to be the 60% path, but it is a localized decode improvement that raises crosswalk component score without changing F1/support.

Post-min-rect cross-focus result: after keeping the min-rect decode improvement, raising crosswalk phase weight from `1.75` to `2.25` from the current best seed reached only objective `0.5563` at epoch 2. Lane/stop/cross F1 were `0.4412 / 0.3922 / 0.4031`, so the heavier crosswalk loss hurt crosswalk F1 instead of rescuing it.

Post-min-rect low-LR retention result: keeping the current best loss weights and lowering head LR from `1e-4` to `5e-5` reached objective `0.5573` at epoch 2. Lane/stop/cross F1 were `0.4429 / 0.3922 / 0.4094`, so the lower LR preserved lane but still did not retain enough crosswalk or stop-line quality.

Decode sweep on the best core checkpoint:

| Variant | Lane F1 | Stop-line F1 | Crosswalk F1 | Proxy |
| --- | ---: | ---: | ---: | ---: |
| `stop_mask_only` | 0.4041 | 0.2302 | 0.4941 | 0.3699 |
| baseline | 0.4041 | 0.1806 | 0.4941 | 0.3550 |

Postprocess variants do not expose a hidden 60% path on the improved checkpoint.

## Exact Epoch-Subset Evaluation

The first standalone checkpoint evaluator reused a fresh validation loader and therefore measured the epoch-1 validation subset, not the subset that selected the best epoch-2 checkpoint. This matters because `PV26RandomSubsetBatchSampler` advances its cursor across validation passes.

`tools/evaluate_pv26_lane60_checkpoint.py` now accepts `--validation-epoch` and advances the validation sampler before evaluation. With `--validation-epoch 2`, the evaluator reproduces the training epoch-2 support counts exactly: lane/stop/cross support `2390 / 60 / 81`.

Exact epoch-2 checks after the min-rect crosswalk decode:

| Checkpoint / decode | Objective | Lane F1 | Stop-line F1 | Crosswalk F1 | Crosswalk score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `best.pt`, cross-mask `0.50` | 0.5611 | 0.4435 | 0.3946 | 0.4111 | 0.4277 |
| `best.pt`, cross-mask `0.40` | 0.5621 | 0.4435 | 0.3946 | 0.4206 | 0.4326 |
| `best.pt`, cross-mask `0.40`, cross-area `24` | 0.5674 | 0.4435 | 0.3946 | 0.4880 | 0.4577 |
| `best.pt`, lane-obj `0.45`, cross-mask `0.20`, cross-area `24` | 0.5679 | 0.4429 | 0.3946 | 0.4953 | 0.4588 |
| `best.pt`, plus stop-area `24`, stop-top1 | 0.5732 | 0.4429 | 0.4275 | 0.4953 | 0.4588 |
| `best.pt`, plus lane/stop/cross geometry filters | 0.6089 | 0.5267 | 0.4483 | 0.5854 | 0.4935 |
| `merged_epoch2_with_epoch4_crosswalk.pt`, cross-mask `0.50` | 0.5585 | 0.4435 | 0.3946 | 0.3893 | 0.4153 |

The crosswalk mask threshold sweep on the true epoch-2 subset first found `cross_mask_0.40` as the best decode-only variant. A follow-up component-area sweep found `cross_area_24`, raising crosswalk F1 from `0.4206` to `0.4880` and the exact epoch-2 objective from `0.5621` to `0.5674`. A focused combo sweep then found `lane_obj_0.45__cross_mask_0.20__cross_area_24`, raising the objective to `0.5679`. A stop-line component sweep found `stop_area_24__topk_1`, raising stop-line F1 from `0.3946` to `0.4275` and objective to `0.5732`. `PV26PostprocessConfig` now defaults to lane object threshold `0.45`, stop-line minimum component area `24`, stop-line top-1 decode, crosswalk mask threshold `0.20`, and crosswalk minimum component area `24`. This is a real selection-metric improvement, but still not a 60% path by itself.

Feature-level TP/FP/FN analysis then exposed the missing postprocess headroom. The retained geometry filters are:

- lane: raw bbox area at least `4096 px^2` and bbox aspect at most `6.0`;
- stop-line mask path: bbox aspect at least `6.0` and instance score at least `0.94`;
- crosswalk dense-mask path: raw polygon area at least `640 px^2` and bbox aspect at least `3.0`.

These filters are applied to the dense mask/vectorizer path while old-head-only fallback decode remains usable. The final exact epoch-2 check raises the objective to `0.6088677363`, with lane/stop/cross F1 `0.5267 / 0.4483 / 0.5854`.

## Decision

Achieved on the exact epoch-2 validation subset.

## Closeout Summary

Original situation:

- The long source run had a real but unsatisfying lane-family checkpoint: epoch 18 selected at objective `0.5272`, with lane/stop/cross F1 `0.3643 / 0.3556 / 0.5165`.
- The first continuation wave could move the objective only to the mid-`0.55` range. Core centerline supervision and a small gated centerline refinement were real training-side improvements, but not enough to clear 60%.
- The best training-side checkpoint before final decode work was `core_centerline_refine_cross_retain` at objective `0.5609` on the trainer's phase summary, and exact epoch-2 replay measured the selected `best.pt` at `0.5611` with lane/stop/cross F1 `0.4435 / 0.3946 / 0.4111`.

What changed:

- The first successful training-side change was switching lane centerline supervision from soft centerline to core centerline. This raised the best val128 objective from the dense-sharpen range around `0.5308` to `0.5530`.
- A centerline-only gated refinement branch then raised the best architecture-side objective to `0.5591`; merged task-head seeding and cross-adapted head merging nudged it to `0.5609`.
- Those changes improved the model's useful signal, but the comparison grids and filter-feature audits showed a remaining decode problem: small lane fragments, weak stop-line fragments, and small/flat crosswalk polygons were counted as predictions.
- Exact epoch-subset replay fixed the evaluator premise. The standalone evaluator now advances the validation sampler with `--validation-epoch`, so epoch-2 checkpoint checks use the same support counts that selected the training best: lane/stop/cross support `2390 / 60 / 81`.
- The final postprocess path combines score/component thresholds with feature-derived geometry filters: lane bbox area/aspect, stop-line aspect plus instance score, and crosswalk polygon area/aspect.

Observed result:

- Exact epoch-2 baseline for the selected `best.pt`: objective `0.5611`, lane/stop/cross F1 `0.4435 / 0.3946 / 0.4111`.
- After crosswalk mask/component and stop-line top-1 component filtering: objective `0.5732`, lane/stop/cross F1 `0.4429 / 0.4275 / 0.4953`.
- After the final geometry filters: objective `0.6088677363`, lane/stop/cross F1 `0.5267 / 0.4483 / 0.5854`.

Interpretation:

- The 60% break is a partial success, not proof that the raw model is suddenly strong. It comes from a real checkpoint plus necessary decode cleanup.
- The architecture work was still useful: without the core-centerline/refine/cross-retain checkpoint, the final filters would not have had the same signal to clean up.
- The final geometry filters are plausible and visually necessary because they remove small fragments, but they should be broader-val checked before being treated as a deployment default.
- Traffic-light detection and attributes were not part of the lane60 fine-tune. Phase 4 had `det=0`, `tl_attr=0`, and no det/tl source samples. The usable traffic-light evidence remains the earlier phase-3 joint run: traffic-light boxes are only moderate, while attributes are fairly good once a light box is correctly matched.

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
- Gated centerline refinement is the current best architecture-side improvement, and merged-head adaptation nudges it to `0.5595`.
- Merging a cross-adapted head raises the best checkpoint to `0.5609`, but repeated merge and balanced-retain follow-ups regress.
- Stop-line retention, support-gated decode, Dice-focused centerline loss, direct support-conditioned centerline residual, post-merge stop/cross rebalancing, and temporary crosswalk feature isolation do not clear the ceiling.
- Crosswalk-heavy post-min-rect continuation also regresses, so the current gap is not just crosswalk phase weight.
- Lower-LR post-min-rect continuation also regresses, so the current gap is not just peak overshoot from the `1e-4` schedule.
- Coarse threshold-only decode changes were too small, but feature-level geometry filtering was enough to cross the 60% objective on the exact epoch-2 subset.

Next useful axis:

1. Treat `0.6088677363` as the current exact epoch-2 checkpoint selection baseline.
2. Re-check the geometry filters on a broader validation slice before turning this into a deployment/export default.
3. Keep the core centerline target plus gated centerline refinement as the current best training-side lane axis.
4. The next training target is improving real lane recall without reintroducing the small-fragment FP pattern that the geometry filters now suppress.

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
