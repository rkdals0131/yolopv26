# temp GPT Pro Handoff: PV26 Lane-Family Architecture Breakthrough Review

> Temporary GPT Pro prompt / handoff.
> Canonical maintained status remains:
> `docs/00A_CURRENT_STATUS.md`, `docs/00B_STATUS_HISTORY.md`, and
> `docs/00C_NEXT_GATES.md`.
>
> This document intentionally overwrites the earlier postprocess-heavy handoff.
> The new request is broader: review whether the blocker is architecture,
> training exposure, head/neck allocation, loss balancing, decoder contract, or
> postprocess. Do not assume the remaining gap is mainly a postprocess problem.

## 1. Actual Objective

The real target is broader validation success:

- Validation protocol: broader-val512, validation epoch `2`, batch size `4`.
- All three lane-family task F1 values must be `>= 0.60`:
  - `lane`
  - `stop_line`
  - `crosswalk`
- `phase_objective > 0.60` is not success.
- Mean task score is not success if any task remains below `0.60`.
- Exact-val128 is an intermediate gate only.
- Tiny/smoke val4 is a cheap rejection gate only.
- Oracle, GT-copy, replay-only, or artifact-only recombination results are
  planning evidence, not production success.

Report every claim with task F1 plus TP/FP/FN and support. F1 alone is too easy
to misread.

## 2. Current Verified Status

Current objective-best broader runtime/postprocess composite:

| Task | F1 | TP / FP / FN | Gap To 0.60 | Status |
| --- | ---: | --- | ---: | --- |
| lane | `0.5628` | `4532 / 2097 / 4945` | `+0.0372` | below target |
| stop_line | `0.4235` | `101 / 105 / 170` | `+0.1765` | main runtime bottleneck |
| crosswalk | `0.6187` | `232 / 123 / 163` | pass | preserve |

Current objective value:

- `0.6230558330631257`

This objective is a training/selection proxy, not the user-facing success
metric.

Known broader task-balance lower bound:

| Task | F1 | TP / FP / FN | Source |
| --- | ---: | --- | --- |
| lane | `0.5628` | `4532 / 2097 / 4945` | current lane/crosswalk runtime composite |
| stop_line | `0.5164` | `126 / 91 / 145` | projection-competition replay |
| crosswalk | `0.6187` | `232 / 123 / 163` | hull crosswalk decode |

Important: `0.5628 / 0.5164 / 0.6187` is not a single checkpoint's raw runtime
output. It is an artifact-only lower bound combining the current lane/crosswalk
runtime composite with stop-line projection-competition replay. It proves useful
headroom, but it is not deployable unless that stop-line replay logic becomes a
real runtime contract.

## 3. Retained Artifacts

Retained base run:

- `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412`
- Checkpoint: `phase_4/checkpoints/best.pt`

Current merged lane-head composite:

- `runs/pv26_exhaustive_od_lane_train/lane60_lane_head_transplant_original_stop_pca_20260512`
- Checkpoint: `merged_lane_head.pt`
- Retained broader metrics:
  - `analysis_exports/lane_task_mask_context_val512_epoch2/metrics.csv`
  - `analysis_exports/lane_task_mask_context_val512_epoch2/summary.json`

Retained exact stop-line lane-extent probe:

- `runs/pv26_exhaustive_od_lane_train/lane60_core_centerline_refine_cross_retain_from_exhaustive_od_lane_default_20260505_032217_default_20260510_003412/analysis_exports/stopline_lane_extent_readout_val128_epoch2/variants.csv`

Older exact-eval and visual-check exports were pruned from active `runs` to
reduce disk usage. Prefer retained files above or regenerate with the tools
listed below.

## 4. Code Map

Training/evaluation entrypoints:

- `tools/run_pv26_lane60_probe.py`
- `tools/evaluate_pv26_lane60_checkpoint.py`
- `tools/probe_pv26_lane_flip_tta.py`
- `tools/probe_pv26_stopline_candidate_pool.py`

Training schedule / scenario logic:

- `tools/pv26_train/scenarios.py`
- `tools/pv26_train/runtime.py`

Unified heads and task-specific heads:

- `model/net/roadmark_v2_heads.py`
- `model/net/lane_head_segfirst.py`
- `model/net/stopline_head_line.py`

Decoder / postprocess / vectorizer:

- `model/engine/postprocess.py`
- `model/engine/lane_segfirst_vectorizer.py`

Loss / metrics:

- `model/engine/loss.py`
- `model/engine/metrics.py`

Current reproducibility tools:

- `tools/probe_pv26_lane_flip_tta.py`
- `tools/probe_pv26_stopline_candidate_pool.py`
- `tools/probe_pv26_lane_instance_evidence.py`
- `tools/probe_pv26_lane_fn_recovery_audit.py`
- `tools/analyze_pv26_lane_repairability_model_replay.py`
- `tools/replay_pv26_lane_point_repair.py`
- `tools/probe_pv26_lane_ranked_translate_repair.py`

## 5. Evaluation Contract Details

Metric shape:

- Lane and stop-line matching are distance-based, using resampled
  polylines/line segments and Hungarian assignment under task-specific
  thresholds.
- Crosswalk is polygon-oriented. Hull decode is currently the retained positive
  path.
- `phase_objective` is a proxy assembled from task F1 and geometry/attribute
  terms. It is not equivalent to minimum task F1.

Required gates for a serious candidate:

1. Smoke val4 rejection gate.
2. Exact val128 gate.
3. Broader val512 gate.

Required output at each gate:

- `lane` F1, TP, FP, FN, support.
- `stop_line` F1, TP, FP, FN, support.
- `crosswalk` F1, TP, FP, FN, support.
- A clear statement of whether the result is raw runtime behavior,
  postprocess-only, replay-only, or oracle/GT-assisted.

## 6. Crosswalk Status

Crosswalk is closest to done:

- Current broader F1: `0.6187`
- TP/FP/FN: `232 / 123 / 163`
- Positive retained path: `crosswalk_polygon_mode=hull`

Do not spend the main effort on crosswalk unless the proposal also preserves or
improves lane and stop-line. Keep hull decode unless there is direct evidence
that a replacement improves broader-val512 without harming the other tasks.

Closed crosswalk families:

- Object/mask/component-area threshold sweeps.
- Polygon-area/aspect/top-k sweeps.
- Exact-val128-only threshold wins.

## 7. Lane Diagnosis

Current broader lane runtime F1:

- `0.5628`
- TP/FP/FN: `4532 / 2097 / 4945`
- Gap to `0.60`: `+0.0372`

What has worked:

- Row-scan vectorizer improved over the earlier component path.
- Tangent-link row-scan improved continuity.
- Segment-MIL lane-head-only transplant became part of the current composite.
- Flip-centerline average helped.
- Fixed crosswalk-mask lane suppression improved `0.5577 -> 0.5628`.

Important headroom evidence:

- GT centerline-core oracle can raise lane much higher, so the blocker is not
  only vectorizer mechanics.
- At current FP, lane F1 `0.60` needs roughly `489` recovered FNs.
- `center>=0.50 and unmatched<=80px` bucket has `563` FNs, enough for a
  no-new-FP upper bound around `0.6062`.
- `unmatched<=120px any center` maps `1698` FN rows to many unmatched
  predictions and implies more oracle headroom.
- Lane repairability ranker is promising as selection evidence:
  - broad label AUC/AP `0.6821 / 0.7491`
  - top-500 oracle-repair lane F1 `0.6077`
  - top-1000 oracle-repair lane F1 `0.6528`
- But geometry repair remains weak, so this is not production success.

Closed lane families. Do not repeat as simple sweeps:

- Scalar centerline threshold calibration.
- Centerline target mode / auxiliary weight-only sweeps.
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
- Simple translation, centerline snap, affine snap, component-row projection,
  and row-profile softargmax repair.
- Ridge/polyline/KNN residual reconstruction from prediction-side features.
- Temporal-neighbor lane union.

Open lane hypothesis class:

- The remaining lane gap may be a representation/instance-separation problem,
  not a threshold problem.
- Consider a dual contract: dense seg-first evidence plus a row/instance-native
  decoder that learns separable lane instances rather than only vectorizing
  thresholded heatmaps.
- Consider whether the layer composition, neck routing, or head capacity limits
  instance stability before postprocess sees the evidence.

## 8. Stop-Line Diagnosis

Current broader stop-line runtime F1:

- `0.4235`
- TP/FP/FN: `101 / 105 / 170`
- Gap to `0.60`: `+0.1765`

Current projection-competition replay reference:

- `0.5164`
- TP/FP/FN: `126 / 91 / 145`
- Gap to `0.60`: `+0.0836`

Important positives:

- Exact val128 candidate-pool regeneration is reproducible:
  - `1170` candidate rows
  - `442` oracle-positive rows
  - baseline exact F1 `0.4483`, TP/FP/FN `26 / 30 / 34`
  - projection-competition exact F1 `0.5167`, TP/FP/FN `31 / 29 / 29`
- Dense signal often exists:
  - val128 stop-line GT `60`
  - GT tube mask max `>=0.50` for `51 / 60`
  - GT tube center max `>=0.50` for `50 / 60`
  - anchorless component fit close for `34 / 60`

Key diagnosis:

- Local candidate geometry error is mostly along the stop-line axis, not normal
  to it.
- Axis-projection plus GT length oracle can reach around `0.6559`
  (`162 / 61 / 109`).
- Full GT-midpoint plus GT-length oracle is only slightly higher, around
  `0.6599`.
- Fixed min-length or predicted-offset variants do not recover the same gain.
- Therefore the missing production signal is no-GT along-axis midpoint shift
  plus extent/length recovery with FP control.

Why selector-only is not enough:

- Projection-competition reference is `0.5164`, TP/FP/FN `126 / 91 / 145`.
- If FP stays fixed, F1 `0.60` needs about `+30` TP.
- Recovering every positive-misrank row alone is almost but not quite enough
  (`0.5996`), so either FP must drop or no-oracle positives must be recovered.

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
- Axis score-profile weighting.
- Fragment-axis auxiliary contract.
- Temporal context gap/frame smoothing on sparse validation candidates.
- Same-axis support span.
- Raw-image axis stripe midpoint/extent.
- Flip consensus/flip union candidate variants.
- Lane-crossing extent readout.
- Task-mask competition strength/source/mask threshold sweeps.
- Proposal-island midpoint and baseline-absent fallback.
- P4 context head and zero-gated coarse context head without a new premise.

Open stop-line hypothesis class:

- The stop-line head may need a different output contract: endpoint, midpoint,
  half-length, and existence probability tied to dense line support, rather
  than relying on mask components plus post-hoc geometry recovery.
- A task-specific stop-line neck or adapter may be necessary if shared features
  under-serve the thin line geometry.
- The problem may be head/neck/loss allocation, not only decoder logic.

## 9. Architecture / Training Hypotheses GPT Pro Should Consider

Do not stay overly conservative. Treat the following as first-class candidate
classes, not as speculative afterthoughts.

### A. Training exposure and freeze schedule

Hypothesis:

- Lane-family heads may be learning too late or under too narrow a freeze
  schedule. If early stages under-expose stop-line/crosswalk/lane-family labels,
  stage 4 may be trying to repair a representation that never became useful for
  these tasks.

Ask GPT Pro to inspect:

- `tools/pv26_train/scenarios.py`
- stage ratios, `task_positive_fraction`, `aihub_lane` exposure
- freeze/unfreeze points
- whether upper trunk/neck layers should reopen during lane-family stages

Useful proposal shape:

- One controlled schedule change only.
- Preserve current best lane/crosswalk decode.
- Compare gradient norms and TP/FP/FN movement against current baseline.

### B. Stop-line segment contract rewrite

Hypothesis:

- Stop-line failure is not solved by selecting better existing candidates. The
  head may need to directly predict an oriented segment contract: existence,
  axis angle, midpoint shift along axis, endpoint coordinates or half-length,
  and uncertainty/confidence.

Ask GPT Pro to distinguish this from:

- Dense Hough sweeps.
- Axis-profile postprocess.
- Candidate-reranking.

Useful proposal shape:

- Model-side or decoder-side single-axis change.
- A small set of segment candidates with calibrated existence probability.
- Explicit no-GT midpoint/extent recovery and FP control.

### C. Task-specific neck / adapter isolation

Hypothesis:

- Stop-line and lane geometry may be degraded by shared feature competition.
  The issue may be feature routing and head/neck allocation, not only loss
  scale or postprocess.

Ask GPT Pro to inspect:

- `model/net/roadmark_v2_heads.py`
- existing stop-line feature-isolation code paths
- whether unified `PV26Heads` actually routes through the intended isolated
  variant
- whether P2/P3/P4/P5 allocation matches each task's geometry scale

Useful proposal shape:

- Activate or minimally wire an existing task-specific neck/adaptor path rather
  than inventing a large architecture from scratch.
- Prove movement in stop-line TP/FN without harming crosswalk hull retention.

### D. Multi-task loss and gradient distribution

Hypothesis:

- Static loss weights or sampler ratios may be starving the hardest geometry
  tasks, or one task may dominate shared representation updates.

Ask GPT Pro to consider literature-backed methods only if tied to measurable
diagnostics:

- Uncertainty weighting.
- GradNorm-style balancing.
- PCGrad-style conflict mitigation.
- Task-specific adapters / cross-stitch / attention sharing.

Useful proposal shape:

- First add a gradient/loss audit if evidence is missing.
- Then change one balancing mechanism.
- Do not propose generic "tune weights" unless it has a falsifiable gate.

### E. Lane dual decoder / instance-native contract

Hypothesis:

- Current lane evidence has recoverable centerline support but poor production
  instance recovery. A stronger lane decoder may need to learn instance
  grouping directly, not only postprocess centerline heatmaps.

Ask GPT Pro to compare:

- Dense seg-first centerline vectorization.
- Row-native lane representations.
- Anchor/query/set-prediction lane decoders.
- Hybrid dense-evidence plus sparse-instance contracts.

Useful proposal shape:

- No GT at runtime.
- Recover centerline-supported FNs with bounded FP.
- Include explicit instance-level FP control.

## 10. Reference Method Families Worth Consulting

Use references to justify mechanisms, not as generic "try transformer" advice.

Relevant families:

- DETR-style set prediction and Hungarian matching for sparse object/segment
  outputs.
- Lane set/query decoders and one-to-many lane assignment methods.
- Line segment representations that predict attraction fields, endpoints, or
  line priors.
- Deep Hough / line-prior methods, but only if they solve the exact stop-line
  axis midpoint/extent failure and are not just another dense Hough repeat.
- Multi-task learning methods for gradient conflict and task feature sharing:
  uncertainty weighting, GradNorm, PCGrad, cross-stitch networks, MTAN, and
  task-adaptive parameter sharing.

GPT Pro should cite specific papers or methods when recommending an
architecture/training change, and explain how the method maps to this repo's
observed TP/FP/FN failure.

## 11. What To Avoid

Avoid generic advice:

- train longer
- use more data
- increase model size
- add attention
- use a transformer decoder
- tune thresholds
- use better augmentation

These are only acceptable if tied to one exact failure mechanism, one changed
axis, concrete file touchpoints, and smoke/exact/broader rejection gates.

Also avoid:

- lane-only training as the default path
- exact-val128 success as final success
- `phase_objective > 0.60` as final success
- treating task-balance replay as deployable checkpoint behavior
- another sweep inside a closed family
- assuming postprocess is the only remaining problem

## 12. Desired GPT Pro Output

Return 3 to 5 ranked breakthrough candidates.

For each candidate, include:

1. Hypothesis
   - What failure mode does it address?
   - Why might it be architecture/training/head/neck/loss/decoder, not merely
     postprocess?
   - Which closed family is it distinct from?

2. Code touchpoints
   - Exact files/functions likely touched.
   - Whether it is model-side, decoder-side, loss-side, sampler-side, schedule
     side, or runtime postprocess.

3. Minimal implementation slice
   - One branch/worktree.
   - One changed axis only.
   - No broad refactor.
   - Preserve crosswalk hull decode unless directly testing crosswalk.

4. Verification gate
   - Smoke val4 rejection gate.
   - Exact val128 gate.
   - Broader val512 gate.
   - Required TP/FP/FN movement.

5. Stop criteria
   - What result closes the idea as negative?
   - What result justifies broadening?

6. Risk
   - Which task might regress?
   - How to preserve current lane/crosswalk retention?

7. Literature anchor
   - Which paper/method family supports the idea?
   - What exact part of that method maps to the observed PV26 failure?

## 13. Suggested Prompt To GPT Pro

Use this handoff plus the codebase. Propose new methodology candidates to get
broader validation lane / stop_line / crosswalk F1 all above `0.60`.

Important constraints:

- Do not propose repeated threshold/TTA/sweep variants listed as closed.
- Do not optimize only `phase_objective`.
- Do not propose lane-only as the main path.
- Treat `0.5628 / 0.5164 / 0.6187` as an artifact-only lower bound, not a
  solved model.
- Prioritize stop-line no-GT along-axis midpoint/extent recovery with FP
  control.
- Prioritize lane no-GT instance recovery/alignment that actually moves
  TP/FP/FN.
- Keep crosswalk hull decode retained unless a proposal explicitly improves it.
- Be willing to recommend architecture, training schedule, head/neck routing,
  and loss/gradient distribution changes if the evidence supports them.
- Give concrete file-level implementation plans and rejection gates.

The best answer is not the safest answer. It should be bold enough to challenge
the current architecture, but still falsifiable in one-axis experiments.
