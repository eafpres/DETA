#!/usr/bin/env bash


set -euo pipefail

DETA_ROOT="$HOME/DETA/DETA"
CHECKPOINT="$DETA_ROOT/exps/public/deta_swin_ft_e8/checkpoint.pth"
IMAGE_DIR="/mnt/c/users/bbate/desktop/inference_images"
OUTPUT_DIR="/mnt/c/users/bbate/desktop/inference_results"
CLASSES="$DETA_ROOT/data/classes.txt"

mkdir -p "$IMAGE_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$DETA_ROOT"

python "$DETA_ROOT/tools/infer_folder.py" \
  --checkpoint "$CHECKPOINT" \
  --image-dir "$IMAGE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --location-iou-threshold 0.4 \
  --classes "$CLASSES" \
  --threshold 0.4 \
  --stable-wait-ms 500 \
  --blank-width 1200 \
  --blank-height 900 \
  --window-x 50 \
  --window-y 50 \
  "$@"
