#!/bin/bash
# Keep a step-tagged copy of each checkpoint so a CLIPScore-vs-steps curve can be
# built later (train_paper_setting.py overwrites one checkpoint dir every N steps).
# Validates the copy (25 aggregators + parseable meta) and retries otherwise.
SAVE=/root/bayes-tmp/afa-assets/output/paper_run
OUT=/root/bayes-tmp/afa-assets/output/paper_run_steps
mkdir -p "$OUT"
echo "[$(date '+%F %T')] snapshot watcher started (pid $$)" >> "$OUT/snapshots.log"
last=""
while true; do
    if [ -f "$SAVE/train_meta.json" ]; then
        step=$(python3 -c "import json;print(json.load(open('$SAVE/train_meta.json'))['step'])" 2>/dev/null)
        if [ -n "$step" ] && [ "$step" != "$last" ]; then
            d="$OUT/step_$step"
            if [ ! -d "$d" ]; then
                cp -r "$SAVE" "$d" 2>/dev/null
                n=$(ls -d "$d"/aggregator_* 2>/dev/null | wc -l)
                if [ "$n" -ge 20 ] && python3 -c "import json;json.load(open('$d/train_meta.json'))" 2>/dev/null; then
                    echo "[$(date '+%F %T')] snapshot step $step ok ($n aggregators, $(du -sh "$d" | cut -f1))" >> "$OUT/snapshots.log"
                    last="$step"
                else
                    rm -rf "$d"
                    echo "[$(date '+%F %T')] snapshot step $step incomplete, will retry" >> "$OUT/snapshots.log"
                fi
            else
                last="$step"
            fi
        fi
    fi
    sleep 60
done
