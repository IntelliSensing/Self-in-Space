#!/bin/bash
set -euo pipefail

# SIS-Bench Evaluation Script
# Usage: bash run_eval.sh [OPTIONS]
#
# Examples:
#   bash run_eval.sh                          # 使用默认参数
#   bash run_eval.sh --max_frames 16          # 使用16帧
#   bash run_eval.sh --tensor_parallel_size 4 # 使用4卡

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${SIS_DATA_ROOT:-${REPO_ROOT}/data}"

# 默认配置
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATA_FILE="${DATA_FILE:-${DATA_ROOT}/SIS-Bench/SIS-Bench.jsonl}"
FRAMES_DIR="${FRAMES_DIR:-${DATA_ROOT}/SIS-Bench/video}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/uav}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"  # 支持调整后的图片尺寸
MAX_TOKENS="${MAX_TOKENS:-128}"          # 思考模型需要更多 tokens (如 2048)
MAX_FRAMES="${MAX_FRAMES:-32}"
IMAGE_SIZE="${IMAGE_SIZE:-}"  # 留空使用像素范围参数（推荐）
BATCH_SIZE="${BATCH_SIZE:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
LORA_PATH="${LORA_PATH:-}"  # 留空则使用 base model
TASK_TYPES="${TASK_TYPES:-}"  # 留空则评估全部任务类型
PROMPT_STYLE="${PROMPT_STYLE:-default}"  # default | motion (匹配训练CoT prompt)

# 设置环境变量
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

# 切换到脚本所在目录
cd "$(dirname "$0")"

echo "============================================"
echo "SIS-Bench Evaluation"
echo "============================================"
echo "Model: $MODEL_ID"
echo "Data file: $DATA_FILE"
echo "Frames dir: $FRAMES_DIR"
echo "Result dir: $RESULT_DIR"
echo "Max tokens: $MAX_TOKENS"
echo "Max frames: $MAX_FRAMES"
echo "Image size: $IMAGE_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Tensor parallel: $TENSOR_PARALLEL_SIZE"
echo "LoRA path: ${LORA_PATH:-none}"
echo "Task types: ${TASK_TYPES:-all}"
echo "Prompt style: $PROMPT_STYLE"
echo "============================================"

# 构建命令参数
CMD_ARGS=(
    --model_id "$MODEL_ID"
    --data_file "$DATA_FILE"
    --frames_dir "$FRAMES_DIR"
    --result_dir "$RESULT_DIR"
    --max_model_len "$MAX_MODEL_LEN"
    --max_tokens "$MAX_TOKENS"
    --max_frames "$MAX_FRAMES"
    --batch_size "$BATCH_SIZE"
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
    --enforce_eager
    --greedy
)

# 如果指定了 IMAGE_SIZE，则添加参数
if [ -n "$IMAGE_SIZE" ]; then
    CMD_ARGS+=(--image_size "$IMAGE_SIZE")
fi

# 如果指定了 LORA_PATH，则添加 LoRA 参数
if [ -n "$LORA_PATH" ]; then
    CMD_ARGS+=(--lora_path "$LORA_PATH")
fi

# 如果指定了 TASK_TYPES，则添加任务类型过滤参数
if [ -n "$TASK_TYPES" ]; then
    CMD_ARGS+=(--task_types $TASK_TYPES)  # 不加引号，让空格分词为多个参数
fi

# Prompt style (default | motion)
if [ "$PROMPT_STYLE" != "default" ]; then
    CMD_ARGS+=(--prompt_style "$PROMPT_STYLE")
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'Dry run command: python -m uav.vqa.run_uav'
    printf ' %q' "${CMD_ARGS[@]}" "$@"
    printf '\n'
    exit 0
fi

python -m uav.vqa.run_uav "${CMD_ARGS[@]}" "$@"
