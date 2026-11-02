#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
  echo 'Usage: bash inference.sh <checkpoint> <val|test> [QD-DETR options...]' >&2
  exit 2
fi
QD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="$(cd -- "$QD_DIR/.." && pwd)"
cd "$REPO_DIR"
PY="${FRAMESEEK_PYTHON:-python}"
resolve() { "$PY" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"; }
CHECKPOINT="$(resolve "$1")"
SPLIT="$2"
shift 2
FEATS="$(resolve "${FEATURE_ROOT:-features/qvhighlights}")"
DATA="$(resolve "${QV_DATA_DIR:-data/qvhighlights}")"
cd "$QD_DIR"
export PYTHONPATH="$QD_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" qd_detr/inference.py \
  --resume "$CHECKPOINT" --eval_split_name "$SPLIT" \
  --eval_path "$DATA/highlight_${SPLIT}_release.jsonl" \
  --v_feat_dirs "$FEATS/slowfast_features" "$FEATS/clip_features" --v_feat_dim 2816 \
  --t_feat_dir "$FEATS/clip_text_features" --t_feat_dim 512 "$@"
