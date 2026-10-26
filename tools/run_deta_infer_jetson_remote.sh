#!/usr/bin/env bash
set -euo pipefail
#
# Resolve project defaults from this checkout while allowing deployment overrides.
#
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${DETA_REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
JETSON_LAUNCHER="${DETA_JETSON_LAUNCHER:-$HOME/run_deta_infer.sh}"
CHECKPOINT="${DETA_CHECKPOINT:-$REPO_DIR/checkpoints/deta_swin_ft_e12/checkpoint.pth}"
SSD_CHECKPOINT="${DETA_SSD_CHECKPOINT:-$REPO_DIR/checkpoints/ssdlite_weld_region/ssdlite_weld_region_best.pt}"
CLASSES="${DETA_CLASSES:-$REPO_DIR/data/classes.txt}"
IMAGE_DIR="${DETA_IMAGE_DIR:-$HOME/Desktop/inference_images}"
OUTPUT_DIR="${DETA_OUTPUT_DIR:-$HOME/Desktop/inference_results}"
#
# Fail before inference with an actionable message when deployment assets are missing.
#
if [[ ! -x "$JETSON_LAUNCHER" ]]; then
  echo "Installed Jetson launcher is missing or not executable: $JETSON_LAUNCHER" >&2
  echo "Run install_deta_inference_deps_jetson_Orin_NX_jetpack6.2_py310.sh first." >&2
  exit 1
fi
for required_file in "$CHECKPOINT" "$SSD_CHECKPOINT" "$CLASSES"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required inference file not found: $required_file" >&2
    exit 1
  fi
done
mkdir -p "$IMAGE_DIR" "$OUTPUT_DIR"
exec "$JETSON_LAUNCHER" \
  --checkpoint "$CHECKPOINT" \
  --ssd-checkpoint "$SSD_CHECKPOINT" \
  --image-dir "$IMAGE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --classes "$CLASSES" \
  "$@"
