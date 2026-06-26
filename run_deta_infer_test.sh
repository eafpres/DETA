#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/DETA/DETA"

python tools/infer_folder.py \
  --checkpoint "$HOME/DETA/DETA/exps/public/deta_swin_ft_e8/checkpoint.pth" \
  --image-dir "/mnt/c/users/bbate/desktop/images" \
  --output-dir "/mnt/c/users/bbate/desktop/inference_results" \
  --location-iou-threshold 0.4 \
  --test-images /"mnt/c/users/bbate/desktop/test_images" \
  --classes "$HOME/DETA/DETA/data/classes.txt" \
  --threshold 0.4 \
  --stable-wait-ms 500\
  --blank-width 1200 \
  --blank-height 900 \
  --window-x 50 \
  --window-y 50 \
  "$@"
