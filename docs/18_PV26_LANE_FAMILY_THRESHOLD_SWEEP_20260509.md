# PV26 Lane-Family Threshold Sweep - 2026-05-09

## Context

The 2026-05-05 PV26 run reached a usable detector/lane-family checkpoint, but the next decision should not be another multi-day run by default. First, separate cheap postprocess calibration headroom from actual training or architecture headroom.

This note records a small CPU smoke sweep on:

- Checkpoint: `runs/pv26_exhaustive_od_lane_train/exhaustive_od_lane_default_20260505_032217/phase_4/checkpoints/best.pt`
- Scenario phase: phase 4, `stage_4_lane_family_finetune`
- Output: `runs/pv26_exhaustive_od_lane_train/exhaustive_od_lane_default_20260505_032217/analysis_exports/lane_family_threshold_sweep_cpu_val8/`
- Command:

```bash
.venv/bin/python tools/sweep_pv26_lane_family_thresholds.py \
  --max-val-batches 8 \
  --thresholds 0.30,0.50,0.70 \
  --device cpu \
  --output-dir runs/pv26_exhaustive_od_lane_train/exhaustive_od_lane_default_20260505_032217/analysis_exports/lane_family_threshold_sweep_cpu_val8
```

The current local environment did not expose CUDA, so this is intentionally a small CPU smoke run. Do not treat the val8 numbers as final threshold defaults.

## Result

| Config | Lane threshold | Stop-line threshold | Crosswalk threshold | Lane F1 | Stop-line F1 | Crosswalk F1 | Phase-4 proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.50 | 0.50 | 0.50 | 0.3366 | 0.0000 | 0.5517 | 0.2786 |
| combined_task_best | 0.70 | 0.30 | 0.50 | 0.3485 | 0.0000 | 0.5517 | 0.2846 |

Task-specific one-axis bests on this tiny sample:

- Lane: threshold 0.70 improved F1 from 0.3366 to 0.3485 by reducing false positives from 176 to 143, while recall dropped from 0.4182 to 0.3939.
- Crosswalk: threshold 0.50 remained best among 0.30, 0.50, 0.70.
- Stop-line: no true positives in this val8 slice; support is too low to infer anything useful from this smoke run.

## Interpretation

Threshold calibration alone does not look like a 60-70% lane-family solution. The lane head shows a small precision-vs-recall calibration lever, but the gain is marginal on val8. Crosswalk appears already close to the current threshold on this sample. Stop-line needs a larger validation slice before interpreting threshold behavior.

The next cheap gate is a GPU val128 or full-val sweep before changing training defaults. If the larger sweep confirms only a small lane gain, the next real improvement path is not more threshold tuning; it should be a resume/fine-tune experiment from `phase_4/best.pt` with one axis changed at a time.

## Next Command

```bash
.venv/bin/python tools/sweep_pv26_lane_family_thresholds.py \
  --max-val-batches 128 \
  --thresholds 0.20,0.30,0.40,0.50,0.60,0.70,0.80 \
  --device auto \
  --output-dir runs/pv26_exhaustive_od_lane_train/exhaustive_od_lane_default_20260505_032217/analysis_exports/lane_family_threshold_sweep_val128
```

Decision gate after val128:

- If lane threshold 0.60-0.80 gives a material F1 lift without hurting crosswalk/stop-line, adopt it for export/eval first.
- If the lift stays small, keep postprocess fixed and run the next training experiment from `phase_4/best.pt`.
- Do not launch another 4-day run until the val128 sweep and a checkpoint-resume fine-tune config are both explicit.
