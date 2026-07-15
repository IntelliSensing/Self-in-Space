#!/bin/bash
# Qwen3-VL-4B LoRA fine-tuning baseline (no motion encoder)
#
# Usage:
#   bash run_train.sh [extra training args]
#
# Common overrides:
#   MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct
#   OUTPUT_DIR=/path/to/output
#   DATASET_USE=sis_motion_54k
#   CUDA_VISIBLE_DEVICES=0,1
#   SWANLAB_ENABLED=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ====== Environment ======
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"
ALLOC_CONF_VALUE="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export PYTORCH_ALLOC_CONF="${ALLOC_CONF_VALUE}"
unset PYTORCH_CUDA_ALLOC_CONF
export SWANLAB_ENABLED="${SWANLAB_ENABLED:-0}"

# ====== Runtime configuration ======
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
DATASET_USE="${DATASET_USE:-sis_motion_54k}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/qwen3vl-4b-lora}"
DS_CONFIG="${DS_CONFIG:-${SCRIPT_DIR}/ds_zero2.json}"
DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-}"

mkdir -p "$OUTPUT_DIR"

RESUME_ARG=()
LATEST_CKPT="$(ls -dt "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1 || true)"
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from: $LATEST_CKPT"
    RESUME_ARG=(--resume_from_checkpoint "$LATEST_CKPT")
else
    echo "No checkpoint found, training from scratch"
fi

DEEPSPEED_CMD=(deepspeed)
if [ -n "$DEEPSPEED_INCLUDE" ]; then
    DEEPSPEED_CMD+=(--include "$DEEPSPEED_INCLUDE")
fi

echo "Training config:"
echo "  MODEL_NAME_OR_PATH=$MODEL_NAME_OR_PATH"
echo "  DATASET_USE=$DATASET_USE"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  DS_CONFIG=$DS_CONFIG"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "  PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF"
echo "  SWANLAB_ENABLED=$SWANLAB_ENABLED"
if [ -n "$DEEPSPEED_INCLUDE" ]; then
    echo "  DEEPSPEED_INCLUDE=$DEEPSPEED_INCLUDE"
fi

# ====== Launch Training ======
"${DEEPSPEED_CMD[@]}" \
    --module uav.train.train_qwen \
    --deepspeed "$DS_CONFIG" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset_use "$DATASET_USE" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --video_max_frames 32 \
    --video_min_frames 8 \
    --video_max_pixels $((512 * 32 * 32)) \
    --video_min_pixels $((128 * 32 * 32)) \
    --video_fps 2 \
    --report_to "none" \
    "${RESUME_ARG[@]}" \
    "$@"
