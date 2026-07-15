#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export SIS_DATA_ROOT="${SIS_DATA_ROOT:-${ROOT_DIR}/data}"
export VIDEOFLOW_CKPT="${VIDEOFLOW_CKPT:-${ROOT_DIR}/checkpoints/VideoFlow/MOF_kitti.pth}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/motion/output/motion-mllm-flow-align-add}"

exec bash "${ROOT_DIR}/motion/run_train.sh" "$@"
