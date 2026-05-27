# temp GPT Pro Handoff: PV26 Lane-Family F1 0.6 Breakthrough Search

> Temporary handoff for GPT Pro review.
> This is intentionally verbose. The canonical maintained surfaces remain:
> `00A_CURRENT_STATUS.md`, `00B_STATUS_HISTORY.md`, and `00C_NEXT_GATES.md`.

## 1. What GPT Pro Should Optimize For

We are trying to make the PV26 lane-family stack pass the real target:

- Broader validation, not exact subset only.
- All three task F1 values must be `>= 0.60`:
  - `lane`
  - `stop_line`
  - `crosswalk`
- `phase_objective` crossing `0.60` is not success.
- A mean score over tasks is not success if any task is still below `0.60`.
- Oracle, GT-copy, replay-only, or artifact-only recombination results are planning evidence, not production success.

The current goal is paused, not complete.

## 2. Current Bottom Line

Current objective-best broader runtime/postprocess composite:

| Task | F1 | Gap To 0.60 | Status |
| --- | ---: | ---: | --- |
| lane | `0.5628` | `+0.0372` | below target |
| stop_line | `0.4235` | `+0.1765` | main bottleneck by objective-best runtime |
| crosswalk | `0.6187` | pass | keep this retained |

Known broader task-balance replay:

| Task | F1 | Gap To 0.60 | Source |
| --- | ---: | ---: | --- |
| lane | `0.5628` | `+0.0372` | flip-centerline average plus fixed crosswalk-mask lane gate |
| stop_line | `0.5164` | `+0.0836` | projection-competition replay |
| crosswalk | `0.6187` | pass | hull crosswalk decode |

Important: `0.5628 / 0.5164 / 0.6187` is not a single checkpoint's raw runtime output. It is an artifact-only task-balance lower bound combining the current lane/crosswalk runtime composite with the projection-competition stop-line replay. It proves there is still useful headroom, but it is not deployable by itself unless that stop-line replay logic becomes a real runtime contract.

## 3. Retained Artifacts And Reproducibility State

Retained checkpoint artifacts:

- Base run:
  - `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412`
  - checkpoint: `phase_4/checkpoints/best.pt`
- Current merged lane-head composite:
  - `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512`
  - checkpoint: `merged_lane_head.pt`
  - retained objective-best metrics:
    - `analysis_exports/lane_task_mask_context_val512_epoch2/metrics.csv`
    - `analysis_exports/lane_task_mask_context_val512_epoch2/summary.json`
- Retained exact stop-line lane-extent probe:
  - `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/stopline_lane_extent_readout_val128_epoch2/variants.csv`

Older exact-eval and visual-check exports were pruned from active `runs` to reduce disk usage. The docs preserve their numeric results. If GPT Pro needs to reason from artifacts, prefer retained files above or regenerate with the tools listed below.

The active `develop` branch has restored reproducibility tooling for:

- Current lane composite replay:
  - `tools/probe_pv26_lane_flip_tta.py`
  - fixed variants include `baseline`, `flip_centerline_avg`, and `flip_centerline_avg_lane_cross_comp050`.
- Stop-line candidate/projection replay:
  - `tools/probe_pv26_stopline_candidate_pool.py`
  - supports `--dataset-root`, `--proposal-min-gap 4`, and `--projection-competition-replay`.
- Lane repair/replay evidence:
  - `tools/probe_pv26_lane_instance_evidence.py`
  - `tools/probe_pv26_lane_fn_recovery_audit.py`
  - `tools/analyze_pv26_lane_repairability_model_replay.py`
  - `tools/replay_pv26_lane_point_repair.py`
  - `tools/probe_pv26_lane_ranked_translate_repair.py`

## 4. Evaluation Contract

Use the broader validation protocol for final claims:

- broader-val512, validation epoch `2`, batch size `4`.
- Exact val128 can be used as an intermediate gate only.
- Tiny/smoke val4 is only a cheap rejection gate.
- Report task F1 plus TP/FP/FN and support. F1 without counts is too easy to misread.

Task metric shape:

- Lane and stop-line matching are distance-based, using resampled polylines/line segments and Hungarian assignment under task-specific thresholds.
- Crosswalk is polygon-oriented; hull decode is currently the retained positive path.
- `phase_objective` is a training selection proxy, not the final user goal.

## 5. Starting Point vs Current Progress

Broader-val512 starting point after first 60% illusion was removed:

| Metric | Start |
| --- | ---: |
| objective | `0.5943438312141003` |
| lane F1 | `0.5101` |
| stop_line F1 | `0.4083` |
| crosswalk F1 | `0.5854` |
| lane TP/FP/FN | `3902 / 1920 / 5575` |
| stop-line TP/FP/FN | `98 / 111 / 173` |
| crosswalk TP/FP/FN | `216 / 127 / 179` |
| support lane/stop/cross | `9477 / 271 / 395` |

Current retained objective-best broader runtime/postprocess composite:

| Metric | Current |
| --- | ---: |
| objective | `0.6230558330631257` |
| lane F1 | `0.5628` |
| stop_line F1 | `0.4235` |
| crosswalk F1 | `0.6187` |
| lane TP/FP/FN | `4532 / 2097 / 4945` |
| stop-line TP/FP/FN | `101 / 105 / 170` |
| crosswalk TP/FP/FN | `232 / 123 / 163` |
| support lane/stop/cross | `9477 / 271 / 395` |

Net progress from start to objective-best:

| Task | F1 Delta | TP Delta | FP Delta | FN Delta |
| --- | ---: | ---: | ---: | ---: |
| lane | `+0.0527` | `+630` | `+177` | `-630` |
| stop_line | `+0.0151` | `+3` | `-6` | `-3` |
| crosswalk | `+0.0333` | `+16` | `-4` | `-16` |

Current task-balance lower bound:

| Task | F1 | TP/FP/FN |
| --- | ---: | --- |
| lane | `0.5628` | `4532 / 2097 / 4945` |
| stop_line | `0.5164` | `126 / 91 / 145` |
| crosswalk | `0.6187` | `232 / 123 / 163` |

Even with task-balance replay, lane still needs `+0.0372` F1 and stop-line still needs `+0.0836` F1.

## 6. Current Runtime/ROS Implication

Do not describe this as a solved ROS deployment model.

What is fair to say:

- `merged_lane_head.pt` plus the current lane/crosswalk postprocess should give plausible lane/crosswalk outputs on Korean road driving frames.
- Crosswalk is the most stable of the three after `crosswalk_polygon_mode=hull`.
- Lane is visibly useful but below target. Its broader F1 is `0.5628`, driven by recall still being low.

What is not fair to say:

- Do not say the model has stop-line production F1 `0.5164` as a raw checkpoint behavior. That value is projection-competition replay.
- Do not say the full lane-family target is achieved.
- Do not assume rosbag playback will reproduce the artifact-only task-balance result unless the exact runtime postprocess/replay contract is implemented in the inference path.

## 7. Architecture And Code Map

Important training/evaluation entrypoints:

- `tools/run_pv26_lane60_probe.py`
- `tools/evaluate_pv26_lane60_checkpoint.py`
- `tools/probe_pv26_lane_flip_tta.py`
- `tools/probe_pv26_stopline_candidate_pool.py`

Important lane code:

- `model/net/lane_head_segfirst.py`
- `model/engine/lane_segfirst_vectorizer.py`
- `model/engine/loss.py`
- `tools/probe_pv26_lane_instance_evidence.py`
- `tools/probe_pv26_lane_fn_recovery_audit.py`
- `tools/analyze_pv26_lane_repairability_model_replay.py`
- `tools/replay_pv26_lane_point_repair.py`
- `tools/probe_pv26_lane_ranked_translate_repair.py`

Important stop-line code:

- `model/net/stopline_head_line.py`
- `model/engine/postprocess.py`
- `model/engine/loss.py`
- `tools/probe_pv26_stopline_candidate_pool.py`
- `tools/probe_pv26_stopline_fragment_projection_competition_readout.py`
- `tools/probe_pv26_stopline_readout_components.py`

Important crosswalk code:

- `model/engine/postprocess.py`
- `tools/evaluate_pv26_lane60_checkpoint.py`
- crosswalk hull decode uses `crosswalk_polygon_mode=hull`.

Important docs:

- `docs/00A_CURRENT_STATUS.md`: current snapshot.
- `docs/00B_STATUS_HISTORY.md`: full history of experiments.
- `docs/00C_NEXT_GATES.md`: next gate and "do not repeat" ledger.

## 8. Crosswalk Status

Crosswalk is currently the task closest to done.

What worked:

- Rectangle/min-area/aspect threshold sweeps did not close broader-val512.
- Representation-aware hull decode did:
  - broader-val512 crosswalk F1 `0.6187`
  - TP/FP/FN `232 / 123 / 163`
- Keep `crosswalk_polygon_mode=hull` in future lane/stop-line work.

What not to repeat:

- Do not repeat crosswalk object/mask/component-area/polygon-area/aspect/top-k threshold sweeps as the main path.
- Do not treat exact-val128 crosswalk threshold wins as broader success.

Breakthrough need:

- Crosswalk is not the main bottleneck. Future methods should preserve it, not spend most effort here unless a proposed change also protects lane and stop-line.

## 9. Lane Status

Current lane best:

- Broader-val512 runtime/postprocess lane F1: `0.5628`.
- TP/FP/FN: `4532 / 2097 / 4945`.
- Gap to `0.60`: `+0.0372`.

Major partial positives:

- Row-scan vectorizer improved broader lane over the earlier component path.
- Tangent-link row-scan improved continuity:
  - broader-val512 lane F1 `0.5407`.
- Segment-MIL lane-head-only transplant improved the lane head enough to become part of the current composite.
- Flip-centerline average improved lane:
  - `0.5577` before fixed crosswalk-mask lane suppression.
- Fixed crosswalk-mask lane suppression improved lane:
  - `0.5577 -> 0.5628`
  - TP/FP/FN `4518 / 2206 / 4959 -> 4532 / 2097 / 4945`

Important lane headroom evidence:

- GT centerline-core oracle can raise lane much higher; the blocker is not only vectorizer mechanics.
- Current broader lane F1 `0.60` needs about `489` recovered FNs at current FP.
- GT-labeled buckets show enough candidate headroom:
  - `center>=0.50 and unmatched<=80px` bucket has `563` FNs, enough for a no-new-FP upper bound around `0.6062`.
  - `unmatched<=120px any center` maps `1698` FN rows to many unmatched predictions and can reach much higher oracle F1 if repaired.
- Lane repairability ranker is positive as selection evidence:
  - broad label AUC/AP `0.6821 / 0.7491`
  - top-500 oracle-repair lane F1 `0.6077`
  - top-1000 oracle-repair lane F1 `0.6528`
- But this is not production success because geometry repair remains weak.

Closed lane families. Do not repeat as simple sweeps:

- Scalar centerline threshold calibration.
- Centerline target mode/aux weight-only sweeps.
- Flip max/union/scale/shift/photometric TTA sweeps.
- Task-mask source/strength/mask-threshold sweeps.
- Semantic vote modes.
- Duplicate suppression by class/type or attribute-agnostic distance.
- Per-sample top-k caps and recall-safety gates.
- Endpoint extension.
- Centerline thinning/skeletonization.
- Component polyfit vectorization.
- Row-scan x smoothing.
- Global/Hungarian row assignment.
- Legacy row-head fallback/union.
- Area-rescue threshold/gating variants without a new FP-control signal.
- Residual centerline-component append/gates/replacement.
- Simple translation, centerline snap, affine snap, component-row projection, row-profile softargmax repair.
- Ridge/polyline/KNN residual reconstruction from prediction-side features.
- Temporal-neighbor lane union.

What a useful lane proposal must do:

- Move actual lane TP/FP/FN on smoke first, then exact val128, then broader val512.
- Be no-GT at runtime.
- Either recover FNs from strong predicted centerline evidence or repair unmatched predictions into matched lanes.
- Include explicit FP-control, because several recall-oriented attempts recovered some TP while adding too much FP.
- Avoid relying on GT-copy/oracle replacement except as an upper-bound diagnostic.

Potential open direction class:

- A stronger instance-level centerline recovery/assignment contract, not another threshold.
- A model-side instance-stability signal that produces separable lane instances before the vectorizer.
- A no-GT geometry-alignment signal stronger than local centerline snap or prediction-feature regression.
- A decoder that can generate missing centerline-supported lanes with bounded FP, rather than only modifying existing tracks.

## 10. Stop-Line Status

Current stop-line objective-best runtime F1:

- `0.4235`
- TP/FP/FN `101 / 105 / 170`
- Gap to `0.60`: `+0.1765`

Current stop-line task-balance replay:

- `0.5164`
- TP/FP/FN `126 / 91 / 145`
- Gap to `0.60`: `+0.0836`

This is the hardest task now.

Major partial positives:

- Projection-competition replay is the best retained broader stop-line reference:
  - broader-val512 `0.5164`, TP/FP/FN `126 / 91 / 145`.
- Exact val128 candidate-pool regeneration is reproducible:
  - `1170` candidate rows
  - `442` oracle-positive rows
  - baseline exact stop-line F1 `0.4483`, TP/FP/FN `26 / 30 / 34`
  - projection-competition exact F1 `0.5167`, TP/FP/FN `31 / 29 / 29`
- Dense signal often exists:
  - val128 stop-line GT `60`
  - GT tube mask max `>=0.50` for `51 / 60`
  - GT tube center max `>=0.50` for `50 / 60`
  - anchorless component fit close for `34 / 60`

Key stop-line diagnosis:

- The local candidate geometry error is mostly along the stop-line axis, not normal to it.
- Axis-projection plus GT length oracle can reach stop-line F1 around `0.6559`, TP/FP/FN `162 / 61 / 109`.
- Full GT-midpoint plus GT-length oracle is only slightly higher around `0.6599`.
- Fixed min-length or predicted-offset variants do not recover the same gain.
- Therefore the missing production signal is no-GT along-axis midpoint shift plus extent/length recovery with FP control.

Why selector-only is not enough:

- Projection-competition stop-line reference is `0.5164`, TP/FP/FN `126 / 91 / 145`.
- If FP stays fixed, F1 `0.60` needs about `+30` TP.
- Recovering every positive-misrank row alone is almost but not quite enough (`0.5996`), so either FP must drop or no-oracle positives must be recovered too.

Closed stop-line families. Do not repeat as simple sweeps:

- Projection-competition feature/rank/threshold sweeps as the main path.
- Candidate row feature/logistic/ridge regressors as direct production geometry.
- Score-island midpoint and linefit.
- Axis-projected predicted-offset readout.
- Axis-profile proposal-cell or offset readout.
- Symmetric axis-profile midpoint forcing.
- Axis-window recenter.
- Dense Hough.
- Scale dense TTA.
- Same-checkpoint teacher-cache self-distill.
- Axis score-profile weighting, which was exact-only and broader-negative.
- Fragment-axis auxiliary contract.
- Temporal context gap/frame smoothing on sparse validation candidates.
- Same-axis support span.
- Raw-image axis stripe midpoint/extent.
- Flip consensus/flip union candidate variants.
- Lane-crossing extent readout.
- Task-mask competition strength/source/mask threshold sweeps.
- Proposal-island midpoint and baseline-absent fallback.
- P4 context head and zero-gated coarse context head without a new premise.

What a useful stop-line proposal must do:

- Change candidate generation or candidate geometry, not just re-rank the same candidates.
- Infer along-axis midpoint and extent/length without GT.
- Include explicit FP-control.
- Prove movement against the projection-competition reference, not only the weaker baseline.
- Report TP/FP/FN and no-oracle recovery, not just average distance.

Potential open direction class:

- Direct dense stop-line segment representation that predicts axis-coordinate endpoints or midpoint/half-length in a way tied to line support.
- A structured decoder that uses dense mask/center/angle support to generate a small set of line candidates with calibrated existence probability.
- A model-side stop-line instance/segment proposal head trained to resolve along-axis extent, not just center heatmap.
- A two-stage contract where dense support creates candidate segments and a learned verifier suppresses FP, but only if the candidate generator recovers no-oracle positives.

## 11. What GPT Pro Should Avoid Suggesting

Avoid generic advice like:

- train longer
- tune thresholds
- use more data
- increase model size
- use better augmentation
- add attention
- use transformer decoder

These are not actionable unless tied to the specific failure evidence above and an exact single-axis experiment design.

Also avoid:

- lane-only training as the default path. The target is always lane + stop_line + crosswalk together unless explicitly changed.
- declaring exact-val128 success as final.
- declaring `phase_objective > 0.60` as final.
- treating task-balance replay as a deployable checkpoint.
- proposing another sweep inside a closed family.

## 12. Desired GPT Pro Output Format

Ask GPT Pro to return 3 to 5 ranked breakthrough candidates. For each candidate, require:

1. Hypothesis:
   - What specific failure mode does it address?
   - Which prior closed family is it distinct from?

2. Code touchpoints:
   - Exact files/functions likely touched.
   - Whether it is model-side, decoder-side, loss-side, sampler-side, or runtime postprocess.

3. Minimal implementation slice:
   - One branch/worktree.
   - One changed axis only.
   - No broad refactor.

4. Verification gate:
   - Smoke val4 rejection gate.
   - Exact val128 gate.
   - Broader val512 gate.
   - Required TP/FP/FN movement.

5. Stop criteria:
   - What result closes the idea as negative?
   - What result justifies broadening?

6. Risk:
   - Which task might regress?
   - How to preserve crosswalk hull and current lane/crosswalk retention?

## 13. Suggested Prompt To GPT Pro

Use this handoff plus the codebase. Propose new methodology candidates to get broader validation lane / stop_line / crosswalk F1 all above `0.60`.

Important constraints:

- Do not propose repeated threshold/TTA/sweep variants listed as closed.
- Do not optimize only `phase_objective`.
- Do not propose lane-only as the main path.
- Treat `0.5628 / 0.5164 / 0.6187` as an artifact-only lower bound, not a solved model.
- Prioritize stop-line no-GT along-axis midpoint/extent recovery with FP control, and lane no-GT instance recovery/repair that actually moves TP/FP/FN.
- Give concrete file-level implementation plans and rejection gates.

## 14. Current Best Next-Step Framing

The strongest remaining direction is not another postprocess sweep. The next real attempt should create a new signal:

- Stop-line: no-GT candidate generation or geometry recovery that solves along-axis midpoint/extent and controls FP.
- Lane: no-GT instance recovery/alignment that converts current missed-centerline/nearby-unmatched headroom into actual TP without FP blow-up.
- Crosswalk: preserve hull decode unless a proposal explicitly improves it without harming lane/stop-line.

The project needs a breakthrough candidate, not another incremental threshold variant.
