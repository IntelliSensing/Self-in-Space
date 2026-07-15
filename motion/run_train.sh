#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
ALLOC_CONF_VALUE="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export PYTORCH_ALLOC_CONF="${ALLOC_CONF_VALUE}"
unset PYTORCH_CUDA_ALLOC_CONF
export SWANLAB_ENABLED="${SWANLAB_ENABLED:-0}"

DS_CONFIG="${DS_CONFIG:-${SCRIPT_DIR}/ds_zero2.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/motion-mllm-flow-align-add}"
PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}}"
VIDEOFLOW_CKPT="${VIDEOFLOW_CKPT:-${REPO_ROOT}/checkpoints/VideoFlow/MOF_kitti.pth}"
CONNECTOR_TYPE="${CONNECTOR_TYPE:-visual_flow}"
DATASET_USE="${DATASET_USE:-sis_motion_54k}"
DEEPSPEED_INCLUDE="${DEEPSPEED_INCLUDE:-}"

if [ ! -f "$VIDEOFLOW_CKPT" ]; then
    echo "VideoFlow checkpoint not found: $VIDEOFLOW_CKPT" >&2
    echo "Download MOF_kitti.pth and place it under checkpoints/VideoFlow/." >&2
    exit 1
fi

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
echo "  PRETRAINED_MODEL_NAME_OR_PATH=$PRETRAINED_MODEL_NAME_OR_PATH"
echo "  VIDEOFLOW_CKPT=$VIDEOFLOW_CKPT"
echo "  CONNECTOR_TYPE=$CONNECTOR_TYPE"
echo "  DATASET_USE=$DATASET_USE"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  DS_CONFIG=$DS_CONFIG"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "  SWANLAB_ENABLED=$SWANLAB_ENABLED"
if [ -n "$DEEPSPEED_INCLUDE" ]; then
    echo "  DEEPSPEED_INCLUDE=$DEEPSPEED_INCLUDE"
fi

cd "$SCRIPT_DIR"
"${DEEPSPEED_CMD[@]}" \
    --module src.uav.train.train_qwen \
    --deepspeed "$DS_CONFIG" \
    --model_type "motion-mllm" \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
    --videoflow_checkpoints_path "$VIDEOFLOW_CKPT" \
    --connector_type "$CONNECTOR_TYPE" \
    --tune_mm_llm False \
    --tune_mm_connector True \
    --tune_mm_vision False \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --dataset_use "$DATASET_USE" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --mm_projector_lr 2e-5 \
    --num_train_epochs 1 \
    --bf16 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --video_max_frames 32 \
    --video_min_frames 8 \
    --video_max_pixels $((512 * 28 * 28)) \
    --video_min_pixels $((128 * 28 * 28)) \
    --video_fps 2 \
    --logging_steps 1 \
    --report_to none \
    "${RESUME_ARG[@]}" \
    "$@"
