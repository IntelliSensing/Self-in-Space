#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-}"

if [[ ! "$PROFILE" =~ ^(eval|lora|motion)$ ]]; then
    echo "Usage: $0 {eval|lora|motion}" >&2
    exit 2
fi

VENV="${ROOT_DIR}/.venv-${PROFILE}"
uv venv --python 3.10 "$VENV"
uv pip sync --python "${VENV}/bin/python" \
    --index-strategy unsafe-best-match \
    "${ROOT_DIR}/environments/uv/${PROFILE}.txt"

if [[ "$PROFILE" != "eval" ]]; then
    uv pip install --python "${VENV}/bin/python" \
        flash-attn==2.7.4 --no-build-isolation
fi

echo "Activate with: source ${VENV}/bin/activate"
