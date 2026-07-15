#!/bin/bash
# Qwen-VL baseline or LoRA evaluation on SIS-Bench
#
# Usage:
#   bash run_eval.sh [extra eval args]
#
# Common overrides:
#   MODEL_PATH=output/qwen3vl-4b-lora
#   RESULT_DIR=results/qwen3vl-4b-lora
#   CUDA_VISIBLE_DEVICES=0
#   DEVICE=auto

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${SIS_DATA_ROOT:-${REPO_ROOT}/data}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/output/qwen3vl-4b-lora}"
DATA_FILE="${DATA_FILE:-${DATA_ROOT}/SIS-Bench/SIS-Bench.jsonl}"
FRAMES_DIR="${FRAMES_DIR:-${DATA_ROOT}/SIS-Bench/video}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/qwen3vl-4b-lora}"
DEVICE="${DEVICE:-auto}"
PROMPT_STYLE="${PROMPT_STYLE:-default}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

echo "Eval config:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  DATA_FILE=$DATA_FILE"
echo "  FRAMES_DIR=$FRAMES_DIR"
echo "  RESULT_DIR=$RESULT_DIR"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "  DEVICE=$DEVICE"
echo "  PROMPT_STYLE=$PROMPT_STYLE"
echo "  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"

CMD=(
python -m src.eval.eval
    --model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --frames_dir "$FRAMES_DIR" \
    --result_dir "$RESULT_DIR" \
    --max_frames 32 \
    --min_frames 8 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --greedy \
    --prompt_style "$PROMPT_STYLE" \
    --device "$DEVICE"
)
CMD+=("$@")

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'Dry run command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

cd "$SCRIPT_DIR"
"${CMD[@]}"

# Example task filter:
# --task_types action_recognition action_recall action_prediction action_sequence path_planning
