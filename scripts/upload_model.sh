#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/model}"
REPO_ID="${1:-}"

if [ -z "$REPO_ID" ]; then
    echo "Usage: $0 <namespace/model-name> [hf upload options]" >&2
    exit 2
fi
shift

REQUIRED_FILES=(
    adapter_config.json
    adapter_model.safetensors
    connector_weights.pt
    MOF_kitti.pth
    sis_motion_config.json
)

if [ ! -f "${MODEL_DIR}/README.md" ]; then
    echo "Warning: ${MODEL_DIR}/README.md is missing; uploading without a model card." >&2
fi

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${MODEL_DIR}/${file}" ]; then
        echo "Missing model file: ${MODEL_DIR}/${file}" >&2
        exit 1
    fi
done

CMD=(hf upload "$REPO_ID" "$MODEL_DIR" . \
    --repo-type model \
    --commit-message "Upload SIS-Motion visual-flow baseline")
CMD+=("$@")

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'Dry run command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"
