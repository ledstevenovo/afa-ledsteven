#!/bin/bash
# Overnight AFA run in the paper's setting: 5 realistic finetunes + JourneyDB.
# Auto-restarts on crash with --resume (step-based checkpoints), stops for good
# at the time limit inside the python script (exit 0) or after MAX_RETRIES.
set -o pipefail
cd /root/bayes-tmp/afa

ASSETS=/root/bayes-tmp/afa-assets
EXPERTS="$ASSETS/models/stable-diffusion-v1-5,\
$ASSETS/models/realistic_vision_v5.1/Realistic_Vision_V5.1.safetensors,\
$ASSETS/models/_new_experts/epicrealism.safetensors"
SAVE="$ASSETS/output/paper_run"
LOG="$ASSETS/logs/paper_run_$(date +%Y%m%d).log"
MAX_RETRIES=5

export PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$SAVE" "$(dirname "$LOG")"

echo "===== AFA paper-setting run $(date '+%F %T') ====="
echo "experts: 3 (sd15, rv, epicrealism) -- best learnable-routing 3-subset (+4.63% vs +5.75% for 5)"
echo "eff batch 8 (2 x accum 4), no grad-ckpt, res 512, lr 1e-4, stop 09:00"
echo "checkpoints: $SAVE | log: $LOG"

RETRY=0
while true; do
    RETRY=$((RETRY + 1))
    python3 -u train_paper_setting.py \
        --model_files "$EXPERTS" \
        --model_save_path "$SAVE" \
        --dataset_json_file "$ASSETS/data/journeydb_data.json" \
        --resolution 512 \
        --batch_size 2 \
        --grad_accum_steps 4 \
        --num_workers 4 \
        --lr 1e-4 \
        --weight_decay 0.01 \
        --lr_warmup_steps 100 \
        --save_every_steps 500 \
        --log_every_steps 10 \
        --eval_every_steps 50 \
        --stop_hour 9 \
        --stop_minute 0 \
        --resume \
        2>&1 | tee -a "$LOG"
    CODE=${PIPESTATUS[0]}
    echo "[$(date +%H:%M:%S)] python exited with $CODE"
    if [ "$CODE" -eq 0 ]; then
        echo "finished normally (time limit or max steps)"; break
    fi
    if [ "$RETRY" -ge "$MAX_RETRIES" ]; then
        echo "giving up after $MAX_RETRIES attempts"; break
    fi
    echo "restarting in 30s with --resume (attempt $RETRY/$MAX_RETRIES)"
    sleep 30
done
echo "===== run ended $(date '+%F %T') ====="
