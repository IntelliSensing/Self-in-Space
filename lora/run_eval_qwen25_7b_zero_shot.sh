#!/bin/bash
# Qwen2.5-VL-7B zero-shot evaluation on SIS-Bench.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/qwen25_7b_zero_shot}"
DEVICE="${DEVICE:-auto}"
PROMPT_STYLE="${PROMPT_STYLE:-strict}"

MODEL_PATH="${MODEL_PATH}" \
RESULT_DIR="${RESULT_DIR}" \
DEVICE="${DEVICE}" \
PROMPT_STYLE="${PROMPT_STYLE}" \
bash "${SCRIPT_DIR}/run_eval.sh" "$@"
