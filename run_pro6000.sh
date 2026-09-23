#!/bin/bash
# AFA training on an RTX PRO 6000 (96 GB) -- or any card with >= 48 GB.
# Differences vs the T4 wrapper (run_paper_setting.sh):
#   * cuDNN ON -- the T4 container's cuDNN is broken (verified 3 ways); a normal box wants it (~1.5-2x)
#   * batch 8-16, no gradient accumulation, no gradient checkpointing
#   * all experts found on disk (5-7) instead of the 3 that fit in 15 GB
#   * more dataloader workers (the GPU is fast enough that 4 workers starve it)
#
# Usage:
#   BATCH=8 bash run_pro6000.sh                    # assets default to ../afa-assets
#   AFA_ASSETS=/data/afa-assets BATCH=8 bash run_pro6000.sh
#   STOP_HOUR=9 bash run_pro6000.sh                # optional time limit
set -o pipefail
cd "$(dirname "$0")"

HERE="$(cd "$(dirname "$0")" && pwd)"
ASSETS="${AFA_ASSETS:-$(dirname "$HERE")/afa-assets}"
BATCH="${BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
WORKERS="${WORKERS:-16}"
SAVE="${SAVE:-$ASSETS/output/paper_run_group1}"
LOG="${LOG:-$ASSETS/logs/paper_run_$(date +%Y%m%d).log}"
MAX_RETRIES="${MAX_RETRIES:-5}"
NUM_LAYERS="${NUM_LAYERS:-2}"
MAX_STEPS="${MAX_STEPS:-0}"
DATASET="${DATASET:-$ASSETS/data/journeydb_data.json}"

# Paper Group I by default; EXPERTS=... overrides (e.g. to add the Group II experts).
EXPERTS="${EXPERTS:-$ASSETS/models/_new_experts/epicrealism.safetensors,\
$ASSETS/models/_new_experts/majicmix_v6.safetensors,\
$ASSETS/models/realistic_vision_v5.1/Realistic_Vision_V5.1.safetensors}"
N=$(( $(echo "$EXPERTS" | tr -cd ',' | wc -c) + 1 ))

STOP_ARGS=""
[ -n "$STOP_HOUR" ] && STOP_ARGS="--stop_hour $STOP_HOUR --stop_minute ${STOP_MINUTE:-0}"
[ "$MAX_STEPS" != "0" ] && STOP_ARGS="$STOP_ARGS --max_optimizer_steps $MAX_STEPS"

# No HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE: from_single_file pulls the CLIP text
# encoder *config* from the hub id openai/clip-vit-large-patch14, so offline mode
# breaks loading .safetensors experts (ValueError/OSError on the first load).
# Prefer the repo venv: the pinned dependency set lives there, not in the base env.
PY="$HERE/.venv/bin/python"
[ -x "$PY" ] || PY=python3

export PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$SAVE" "$(dirname "$LOG")"

echo "===== AFA / RTX PRO 6000 run $(date '+%F %T') ====="
echo "experts ($N): $EXPERTS"
echo "batch $BATCH x accum $GRAD_ACCUM = eff $((BATCH * GRAD_ACCUM)) | workers $WORKERS | cuDNN on | no grad-ckpt"
echo "aggregator layers $NUM_LAYERS | max steps ${MAX_STEPS:-unlimited}"
echo "dataset: $DATASET"
echo "checkpoints: $SAVE | log: $LOG | stop: ${STOP_HOUR:-none}"

RETRY=0
while true; do
    RETRY=$((RETRY + 1))
    "$PY" -u train_paper_setting.py \
        --model_files "$EXPERTS" \
        --st_model_file "$ASSETS/models/stable-diffusion-v1-5" \
        --dataset_json_file "$DATASET" \
        --resolution 512 \
        --batch_size "$BATCH" \
        --grad_accum_steps "$GRAD_ACCUM" \
        --num_workers "$WORKERS" \
        --aggregator_num_layers "$NUM_LAYERS" \
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
