#!/bin/bash
# Quick status of the AFA paper-setting run (safe to run from a phone via ssh).
ASSETS="${AFA_ASSETS:-$(dirname "$(cd "$(dirname "$0")" && pwd)")/afa-assets}"
LOG=$(ls -t $ASSETS/logs/paper_run_*.log 2>/dev/null | head -1)
echo "=== AFA paper-setting run ==="
echo "time: $(date '+%F %T')"
ps -eo pid,etime,cmd | grep "[t]rain_paper_setting" | head -1 || echo "process: NOT RUNNING"
echo
if [ -n "$LOG" ]; then
  echo "log: $LOG"
  echo "--- last routing/learning signals ---"
  grep -aE "^step " "$LOG" | tail -6
  echo "--- last checkpoint ---"
  python3 - <<PY 2>/dev/null
import json, os
p = "$ASSETS/output/paper_run/train_meta.json"
if os.path.exists(p):
    m = json.load(open(p))
    print(f"  step={m['step']} epoch={m['epoch']} saved_at={m['time']}")
    for h in m.get('history', [])[-4:]:
        print(f"  step {h['step']:6d} eval_loss={h['eval_loss']:.5f} eval_w0={h['eval_w0']:.3f} eval_dev={h['eval_dev']:.3f}")
else:
    print("  (no checkpoint yet)")
PY
  echo "--- last 3 log lines ---"
  tail -3 "$LOG"
fi
echo
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
df -h "$ASSETS" | tail -1
