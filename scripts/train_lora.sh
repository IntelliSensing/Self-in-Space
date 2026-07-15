#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SIS_DATA_ROOT="${SIS_DATA_ROOT:-${ROOT_DIR}/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/lora/output/qwen3vl-4b-lora}"

exec bash "${ROOT_DIR}/lora/run_train.sh" "$@"
