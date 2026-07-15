#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${SIS_DATA_ROOT:-${REPO_ROOT}/data}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

MODEL_DIR_OVERRIDE="${MODEL_DIR:-}"
MODEL_DIR="${MODEL_DIR_OVERRIDE:-${SCRIPT_DIR}/output/motion-mllm-flow-align-add}"
LOCAL_MODEL_DIR="${REPO_ROOT}/model"
if [ -n "${MODEL_PATH:-}" ]; then
    MODEL_PATH="${MODEL_PATH}"
elif [ -n "$MODEL_DIR_OVERRIDE" ]; then
    LATEST_CKPT="$(ls -dt "$MODEL_DIR"/checkpoint-* 2>/dev/null | head -1 || true)"
    MODEL_PATH="${LATEST_CKPT:-$MODEL_DIR}"
elif [ -f "${LOCAL_MODEL_DIR}/adapter_config.json" ]; then
    MODEL_PATH="${LOCAL_MODEL_DIR}"
else
    LATEST_CKPT="$(ls -dt "$MODEL_DIR"/checkpoint-* 2>/dev/null | head -1 || true)"
    MODEL_PATH="${LATEST_CKPT:-$MODEL_DIR}"
fi
DATA_FILE="${DATA_FILE:-${DATA_ROOT}/SIS-Bench/SIS-Bench.jsonl}"
FRAMES_DIR="${FRAMES_DIR:-${DATA_ROOT}/SIS-Bench/video}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/uav-flow-align-add}"
DEVICE="${DEVICE:-auto}"
CONNECTOR_TYPE="${CONNECTOR_TYPE:-}"
VIDEOFLOW_CKPT="${VIDEOFLOW_CKPT:-}"
if [ -z "$VIDEOFLOW_CKPT" ] && [ -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH/MOF_kitti.pth" ]; then
    candidate="${REPO_ROOT}/checkpoints/VideoFlow/MOF_kitti.pth"
    if [ -f "$candidate" ]; then
        VIDEOFLOW_CKPT="$candidate"
    fi
fi

echo "Eval config:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  DATA_FILE=$DATA_FILE"
echo "  FRAMES_DIR=$FRAMES_DIR"
echo "  RESULT_DIR=$RESULT_DIR"
echo "  CONNECTOR_TYPE=${CONNECTOR_TYPE:-auto}"
echo "  VIDEOFLOW_CKPT=${VIDEOFLOW_CKPT:-auto}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "  DEVICE=$DEVICE"

CMD=(python -m src.eval.eval \
    --model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --frames_dir "$FRAMES_DIR" \
    --result_dir "$RESULT_DIR" \
    --max_frames 32 \
    --min_frames 8 \
    --max_new_tokens 128 \
    --greedy \
    --prompt_style default \
    --device "$DEVICE")

if [ -n "$CONNECTOR_TYPE" ]; then
    CMD+=(--connector_type "$CONNECTOR_TYPE")
fi
if [ -n "$VIDEOFLOW_CKPT" ]; then
    CMD+=(--videoflow_ckpt "$VIDEOFLOW_CKPT")
fi
CMD+=("$@")

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'Dry run command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

cd "$SCRIPT_DIR"
"${CMD[@]}"
