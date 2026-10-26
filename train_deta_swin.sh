#!/usr/bin/env bash
set -euo pipefail
#
# Resolve defaults from this checkout so Windows drive letters are not embedded in Git.
#
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${DETA_TRAIN_VENV:-$HOME/venvs/deta_train_py312}"
CUDA_HOME_DIR="${CUDA_HOME:-/usr/local/cuda-12.8}"
TORCH_LIB_DIR="${DETA_TORCH_LIB_DIR:-$VENV_DIR/lib/python3.12/site-packages/torch/lib}"
DATA_ROOT="${DETA_TRAIN_DATA_ROOT:-$REPO_DIR/data}"
IMAGE_DIR="${DETA_TRAIN_IMAGES:-$DATA_ROOT/images}"
TRAIN_JSON="${DETA_TRAIN_JSON:-$DATA_ROOT/annotations/DETA_train_mosaics.json}"
VAL_JSON="${DETA_VAL_JSON:-$DATA_ROOT/annotations/DETA_val_mosaics.json}"
OUTPUT_ROOT="${DETA_OUTPUT_ROOT:-$REPO_DIR/exps/public}"
TOTAL_GPUS="${DETA_GPUS:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$TOTAL_GPUS}"
RUN_DISTRIBUTED="$REPO_DIR/tools/run_dist_launch.sh"
CONFIG_SCRIPT="$REPO_DIR/configs/deta_swin_ft.sh"
#
# Validate user input before creating an experiment directory or starting workers.
#
if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 <experiment-suffix> <epochs> <batch-size> [extra DETA args...]" >&2
  echo "Example: $0 8 12 1 --amp" >&2
  exit 2
fi
EXP_NUM="$1"
EPOCHS="$2"
BATCH_SIZE="$3"
shift 3
if [[ ! "$EXP_NUM" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Experiment suffix contains unsupported characters: $EXP_NUM" >&2
  exit 2
fi
for numeric_value in "$EPOCHS" "$BATCH_SIZE" "$TOTAL_GPUS" "$GPUS_PER_NODE"; do
  if [[ ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Epochs, batch size, and GPU counts must be positive integers." >&2
    exit 2
  fi
done
if (( TOTAL_GPUS % GPUS_PER_NODE != 0 )); then
  echo "DETA_GPUS must be evenly divisible by GPUS_PER_NODE." >&2
  exit 2
fi
EXP_NAME="deta_swin_ft_e${EXP_NUM}"
EXP_DIR="$OUTPUT_ROOT/$EXP_NAME"
LOG_FILE="$EXP_DIR/train.log"
#
# Fail with the missing path before activating CUDA or launching distributed workers.
#
for required_file in "$VENV_DIR/bin/activate" "$RUN_DISTRIBUTED" "$CONFIG_SCRIPT" "$TRAIN_JSON" "$VAL_JSON"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required training file not found: $required_file" >&2
    exit 1
  fi
done
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Training image directory not found: $IMAGE_DIR" >&2
  exit 1
fi
#
# Activate the designated project environment so nested launch scripts use its Python.
#
export PYTHONNOUSERSITE=1
export CUDA_HOME="$CUDA_HOME_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
mkdir -p "$EXP_DIR"
#
# Keep caller arguments last so experiments can override the documented defaults.
#
GPUS_PER_NODE="$GPUS_PER_NODE" "$RUN_DISTRIBUTED" "$TOTAL_GPUS" "$CONFIG_SCRIPT" \
  --output_dir "$EXP_DIR" \
  --coco_path "$DATA_ROOT" \
  --coco_train_images "$IMAGE_DIR" \
  --coco_train_ann "$TRAIN_JSON" \
  --coco_val_images "$IMAGE_DIR" \
  --coco_val_ann "$VAL_JSON" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_queries 300 \
  "$@" \
  > "$LOG_FILE" 2>&1
