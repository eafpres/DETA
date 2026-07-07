#!/usr/bin/env bash
set -uo pipefail
BASE_DIR="$HOME/DETA/DETA/exps/public"
EVAL_SCRIPT="$HOME/DETA/DETA/tools/eval_confusion_matrix.py"
COCO_PATH="/mnt/c/damage_model/data"
COCO_VAL_IMAGES="/mnt/x/common_images"
COCO_VAL_ANN="/mnt/x/common_images/temp_images/val/annotations_filtered.json"
DEVICE="cuda"
IOU_THRESHOLD="0.5"
RUN_IDS=(1 2 3 4 5 6 7 8 9 10 11 13 14 15 17 18 19 20 21 22 23 24 25 26)
failures=0
for run_id in "${RUN_IDS[@]}"; do
  exp_name="damages_deta_swin_ft_e${run_id}"
  exp_dir="${BASE_DIR}/${exp_name}"
  eval_dir="${BASE_DIR}/${exp_name}_eval"
  if [[ ! -d "$exp_dir" ]]; then
    echo "missing experiment directory: $exp_dir"
    failures=$((failures + 1))
    continue
  fi
  mkdir -p "$eval_dir"
  mapfile -t checkpoints < <(find "$exp_dir" -maxdepth 1 -type f -name "checkpoint*.pth" | sort -V)
  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "no checkpoints found: $exp_dir"
    failures=$((failures + 1))
    continue
  fi
  for checkpoint_path in "${checkpoints[@]}"; do
    checkpoint_file="$(basename "$checkpoint_path")"
    checkpoint_name="${checkpoint_file%.pth}"
    out_dir="${eval_dir}/${checkpoint_name}"
    mkdir -p "$out_dir"
    echo "evaluating: $checkpoint_path"
    echo "output: $out_dir"
    if python "$EVAL_SCRIPT" \
      --checkpoint "$checkpoint_path" \
      --output-dir "$out_dir" \
      --coco-path "$COCO_PATH" \
      --coco-val-images "$COCO_VAL_IMAGES" \
      --coco-val-ann "$COCO_VAL_ANN" \
      --optimize-thresholds \
      --iou-threshold "$IOU_THRESHOLD" \
      --device "$DEVICE" \
      2>&1 | tee "$out_dir/eval_confusion_matrix.log"; then
      echo "finished: ${exp_name}/${checkpoint_name}"
    else
      echo "failed: ${exp_name}/${checkpoint_name}"
      failures=$((failures + 1))
    fi
  done
done
if [[ "$failures" -gt 0 ]]; then
  echo "completed with failures: $failures"
  exit 1
fi
echo "completed all evaluations"
