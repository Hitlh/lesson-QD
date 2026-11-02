#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${FRAMESEEK_PYTHON:-python}" "$PROJECT_DIR/start_backend.py" "$@"
