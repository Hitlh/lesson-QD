#!/usr/bin/env bash
set -euo pipefail
QD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="$(cd -- "$QD_DIR/.." && pwd)"
cd "$REPO_DIR"
PY="${FRAMESEEK_PYTHON:-python}"
resolve() { "$PY" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"; }
FEATS="$(resolve "${FEATURE_ROOT:-features/qvhighlights}")"
DATA="$(resolve "${QV_DATA_DIR:-data/qvhighlights}")"
RESULTS="$(resolve "${RESULTS_ROOT:-outputs/qvhighlights}")"
cd "$QD_DIR"
export PYTHONPATH="$QD_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" qd_detr/train.py \
  --dset_name hl --ctx_mode video_tef \
  --train_path "$DATA/highlight_train_release.jsonl" \
  --eval_path "$DATA/highlight_val_release.jsonl" --eval_split_name val \
  --v_feat_dirs "$FEATS/slowfast_features" "$FEATS/clip_features" --v_feat_dim 2816 \
  --t_feat_dir "$FEATS/clip_text_features" --t_feat_dim 512 --bsz 32 \
  --results_root "$RESULTS" --exp_id "${EXP_ID:-reproduction}" "$@"
