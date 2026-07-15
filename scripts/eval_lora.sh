#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${ROOT_DIR}/scripts/data_registry.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export SIS_DATA_ROOT="${SIS_DATA_ROOT:-${ROOT_DIR}/data}"
BENCHMARK="${BENCHMARK:-sis_bench}"
export DATA_FILE="${DATA_FILE:-$("$PYTHON_BIN" "$REGISTRY" get "bench.${BENCHMARK}.annotation")}"
export FRAMES_DIR="${FRAMES_DIR:-$("$PYTHON_BIN" "$REGISTRY" get "bench.${BENCHMARK}.media")}"
export RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/lora/results/qwen3vl-4b-lora}"

exec bash "${ROOT_DIR}/lora/run_eval.sh" "$@"
