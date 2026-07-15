#!/bin/bash
# Batch evaluation script for open-source VLMs on SIS-Bench.
# Usage: nohup bash batch_eval.sh > batch_eval.log 2>&1 &
#
# Features:
#   - Background pre-download: later models download while the current model evaluates
#   - Resume support: already-evaluated models are skipped by run_eval.sh
#   - Fail-safe: one model failure doesn't stop the rest

set -o pipefail

MODELS=(
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"


echo "========================================"
echo "Batch SIS-Bench Evaluation"
echo "Started at: $(date)"
echo "Total models: ${#MODELS[@]}"
echo "========================================"

# -------------------------------------------------------------------
# Background download: sequentially pre-download all models.
# Already-cached models return instantly. This runs in parallel with
# the evaluation loop so that by the time we need model N, it's ready.
# -------------------------------------------------------------------
(
    for model_id in "${MODELS[@]}"; do
        echo "[$(date)] [DOWNLOAD] Checking: $model_id"
        hf download "$model_id" > /dev/null 2>&1
        echo "[$(date)] [DOWNLOAD] Ready: $model_id"
    done
) &
DOWNLOAD_PID=$!
echo "[$(date)] Background download process started (PID: $DOWNLOAD_PID)"

# Kill background download on script exit (e.g. Ctrl-C)
trap "kill $DOWNLOAD_PID 2>/dev/null; wait $DOWNLOAD_PID 2>/dev/null" EXIT

# -------------------------------------------------------------------
# Evaluation loop
# -------------------------------------------------------------------
PASSED=0
FAILED=0

for model_id in "${MODELS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "[$(date)] Starting: $model_id"
    echo "----------------------------------------"

    # Per-model overrides (tensor parallelism, context length, output length)
    tp_override=""
    mml_override=""
    mt_override=""
    case "$model_id" in
        Qwen/Qwen2.5-VL-7B-Instruct) tp_override=2 ;;
    esac

    env_vars="MODEL_ID=$model_id"
    if [ -n "$tp_override" ]; then
        echo "  [INFO] TENSOR_PARALLEL_SIZE=$tp_override for $model_id"
        env_vars="$env_vars TENSOR_PARALLEL_SIZE=$tp_override"
    fi
    if [ -n "$mml_override" ]; then
        echo "  [INFO] MAX_MODEL_LEN=$mml_override for $model_id"
        env_vars="$env_vars MAX_MODEL_LEN=$mml_override"
    fi
    if [ -n "$mt_override" ]; then
        echo "  [INFO] MAX_TOKENS=$mt_override for $model_id"
        env_vars="$env_vars MAX_TOKENS=$mt_override"
    fi

    eval "$env_vars bash \"$SCRIPT_DIR/run_eval.sh\""
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] DONE: $model_id"
        PASSED=$((PASSED + 1))
    else
        echo "[$(date)] FAILED (exit $exit_code): $model_id"
        FAILED=$((FAILED + 1))
    fi
done

# Wait for any remaining downloads to finish
wait $DOWNLOAD_PID 2>/dev/null

echo ""
echo "========================================"
echo "Batch Evaluation Complete"
echo "Finished at: $(date)"
echo "Passed: $PASSED / ${#MODELS[@]}"
echo "Failed: $FAILED / ${#MODELS[@]}"
echo "========================================"
