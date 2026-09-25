#!/usr/bin/env bash
# v2 long run: train (resuming after crashes), then tune decoding on the dev half
# and report the held-out test half for each saved role and for the start model.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN=runs/20260925_v2_full_360k
CONFIG=config/pv26_full_data_360k_v2.yaml
DEV=runs/20260925_v2_index
TEST=runs/20260925_val_test_index
START_RUN=runs/20260924_stop_positive_soft_0p5_12k
MAX_RESTARTS=20

state() {  # prints: done | signal | incomplete
  $PY - "$RUN" <<PYEOF
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
try:
    summary = json.loads((run / "summary.json").read_text())
    config = json.loads((run / "run_config.json").read_text())
except (OSError, ValueError):
    print("incomplete"); sys.exit()
if summary.get("stopped_by_signal"):
    print("signal")
elif summary.get("global_step", 0) >= config["train"]["max_steps"] and summary.get("full_validation") is not None:
    print("done")
else:
    print("incomplete")
PYEOF
}

launched=0
if [ ! -f "$RUN/run_config.json" ]; then
  launched=1
  $PY tools/run_pv26_train.py --config "$CONFIG" --output-dir "$RUN" 2>&1 | tee -a "$RUN.log"
fi
restarts=0
while :; do
  s=$(state)
  echo "[longrun] $(date "+%F %T") state=$s restarts=$restarts" | tee -a "$RUN.log"
  [ "$s" = done ] && break
  # A signal stop during this invocation ends it; rerunning this script resumes.
  if [ "$s" = signal ] && [ "$launched" = 1 ]; then echo "[longrun] stopped by signal; resume with: $0" | tee -a "$RUN.log"; exit 0; fi
  [ "$restarts" -ge "$MAX_RESTARTS" ] && { echo "[longrun] too many restarts" | tee -a "$RUN.log"; exit 1; }
  [ "$launched" = 1 ] && { restarts=$((restarts + 1)); sleep 30; }
  launched=1
  $PY tools/run_pv26_train.py --resume-run "$RUN" 2>&1 | tee -a "$RUN.log"
done

for role in best_stop_line best_roadmark latest best; do
  [ -f "$RUN/checkpoints/$role.pt" ] || continue
  $PY tools/tune_pv26_roadmark_decode.py --run "$RUN" --role "$role" \
      --dev-index-run "$DEV" --test-index-run "$TEST" 2>&1 | tee -a "$RUN.log"
done
$PY tools/tune_pv26_roadmark_decode.py --run "$START_RUN" --role best \
    --dev-index-run "$DEV" --test-index-run "$TEST" 2>&1 | tee -a "$RUN.log"
for role in best latest; do
  [ -f "$RUN/checkpoints/$role.pt" ] || continue
  $PY tools/run_pv26_train.py --resume-run "$RUN" --evaluate-only "$role" \
      --eval-index-run "$TEST" --eval-output "test_half_$role.json" 2>&1 | tail -n 3 | tee -a "$RUN.log"
done
echo "[longrun] $(date "+%F %T") finished" | tee -a "$RUN.log"
