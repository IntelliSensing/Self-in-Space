#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-}"

if [[ ! "$PROFILE" =~ ^(eval|lora|motion)$ ]]; then
    echo "Usage: $0 {eval|lora|motion}" >&2
    exit 2
fi

ENV_FILE="${ROOT_DIR}/environments/conda/${PROFILE}.yml"
conda env create --file "$ENV_FILE"

if [[ "$PROFILE" != "eval" ]]; then
    conda run --name "sis-motion-${PROFILE}" \
        python -m pip install flash-attn==2.7.4 --no-build-isolation
fi
