#!/bin/bash
# AFA training on an RTX PRO 6000 (96 GB) -- or any card with >= 48 GB.
# Differences vs the T4 wrapper (run_paper_setting.sh):
#   * cuDNN ON -- the T4 container's cuDNN is broken (verified 3 ways); a normal box wants it (~1.5-2x)
#   * batch 8-16, no gradient accumulation, no gradient checkpointing
#   * all experts found on disk (5-7) instead of the 3 that fit in 15 GB
#   * more dataloader workers (the GPU is fast enough that 4 workers starve it)
#
# Usage:
#   AFA_ASSETS=/data/afa-assets BATCH=8 bash run_pro6000.sh
#   STOP_HOUR=9 AFA_ASSETS=... bash run_pro6000.sh        # optional time limit
set -o pipefail
cd "$(dirname "$0")"

ASSETS="${AFA_ASSETS:-/data/afa-assets}"
BATCH="${BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
WORKERS="${WORKERS:-16}"
SAVE="${SAVE:-$ASSETS/output/paper_run_pro6000}"
LOG="$ASSETS/logs/paper_run_$(date +%Y%m%d).log"
MAX_RETRIES="${MAX_RETRIES:-5}"

EXPERTS="$ASSETS/models/stable-diffusion-v1-5"
EXPERTS="$EXPERTS,$ASSETS/models/realistic_vision_v5.1/Realistic_Vision_V5.1.safetensors"
for f in epicrealism absolute_reality majicmix_v6 realcartoon_realistic; do
    [ -f "$ASSETS/models/_new_experts/$f.safetensors" ] && \
        EXPERTS="$EXPERTS,$ASSETS/models/_new_experts/$f.safetensors"
done
N=$(( $(echo "$EXPERTS" | tr -cd ',' | wc -c) + 1 ))

STOP_ARGS=""
[ -n "$STOP_HOUR" ] && STOP_ARGS="--stop_hour $STOP_HOUR --stop_minute ${STOP_MINUTE:-0}"

export PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$SAVE" "$(dirname "$LOG")"

echo "===== AFA / RTX PRO 6000 run $(date '+%F %T') ====="
echo "experts ($N): $EXPERTS"
echo "batch $BATCH x accum $GRAD_ACCUM = eff $((BATCH * GRAD_ACCUM)) | workers $WORKERS | cuDNN on | no grad-ckpt"
echo "checkpoints: $SAVE | log: $LOG | stop: ${STOP_HOUR:-none}"

RETRY=0
while true; do
    RETRY=$((RETRY + 1))
    python3 -u train_paper_setting.py \
        --model_files "$EXPERTS" \
        --st_model_file "$ASSETS/models/stable-diffusion-v1-5" \
        --dataset_json_file "$ASSETS/data/journeydb_data.json" \
        --resolution 512 \
        --batch_size "$BATCH" \
        --grad_accum_steps "$GRAD_ACCUM" \
        --num_workers "$WORKERS" \
        --lr 1e-4 \
        --weight_decay 0.01 \
        --lr_warmup_steps 100 \
        --save_every_steps 500 \
        --log_every_steps 10 \
        --eval_every_steps 50 \
        --cudnn on \
        --allow_tf32 \
        --resume \
        --model_save_path "$SAVE" \
        $STOP_ARGS \
        2>&1 | tee -a "$LOG"
    CODE=${PIPESTATUS[0]}
    echo "[$(date +%H:%M:%S)] python exited with $CODE"
    [ "$CODE" -eq 0 ] && { echo "finished normally"; break; }
    [ "$RETRY" -ge "$MAX_RETRIES" ] && { echo "giving up after $RETRY attempts"; break; }
    echo "restarting in 30s with --resume (attempt $RETRY/$MAX_RETRIES)"
    sleep 30
done
echo "===== run ended $(date '+%F %T') ====="
