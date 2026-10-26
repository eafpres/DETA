#!/usr/bin/env bash
set -euo pipefail
#
# Resolve data and output defaults from this checkout while supporting other hosts.
#
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${DETA_TRAIN_VENV:-$HOME/venvs/deta_train_py312}"
PYTHON_BIN="$VENV_DIR/bin/python"
CUDA_HOME_DIR="${CUDA_HOME:-/usr/local/cuda-12.8}"
TORCH_LIB_DIR="${DETA_TORCH_LIB_DIR:-$VENV_DIR/lib/python3.12/site-packages/torch/lib}"
DATA_ROOT="${DETA_SSD_DATA_ROOT:-$REPO_DIR/data}"
TRAIN_JSON="${DETA_SSD_TRAIN_JSON:-$DATA_ROOT/annotations/SSD_train_with_good.json}"
VAL_JSON="${DETA_SSD_VAL_JSON:-$DATA_ROOT/annotations/SSD_val_with_good.json}"
IMAGE_DIR="${DETA_SSD_IMAGES:-$DATA_ROOT/images}"
OUTPUT_DIR="${DETA_SSD_OUTPUT_DIR:-$REPO_DIR/checkpoints/ssdlite_weld_region}"
TRAINER="$REPO_DIR/util/train_ssdlite_weld_region.py"
#
# Validate every dependency before creating output or allocating CUDA resources.
#
for required_file in "$PYTHON_BIN" "$TRAINER" "$TRAIN_JSON" "$VAL_JSON"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required SSD training file not found: $required_file" >&2
    exit 1
  fi
done
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "SSD training image directory not found: $IMAGE_DIR" >&2
  exit 1
fi
#
# Isolate the project environment and keep generated checkpoints outside Git tracking.
#
export PYTHONNOUSERSITE=1
export CUDA_HOME="$CUDA_HOME_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"
cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"
#
# Keep caller arguments last so training runs can override these established defaults.
#
exec "$PYTHON_BIN" "$TRAINER" \
  --train-json "$TRAIN_JSON" \
  --val-json "$VAL_JSON" \
  --images-dir "$IMAGE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 30 \
  --batch-size 12 \
  --val-batch-size 4 \
  --lr 1e-4 \
  --weight-decay 5e-2 \
  --num-workers 8 \
  --device cuda \
  "$@"
