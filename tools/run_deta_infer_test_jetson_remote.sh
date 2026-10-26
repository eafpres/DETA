#!/usr/bin/env bash
set -euo pipefail
#
# Layer test-mode settings over the shared launcher to keep setup in one place.
#
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_LAUNCHER="${DETA_REMOTE_LAUNCHER:-$SCRIPT_DIR/run_deta_infer_jetson_remote.sh}"
TEST_IMAGE_DIR="${DETA_TEST_IMAGE_DIR:-$HOME/Desktop/test_images}"
if [[ ! -x "$REMOTE_LAUNCHER" ]]; then
  echo "Jetson inference launcher is missing or not executable: $REMOTE_LAUNCHER" >&2
  exit 1
fi
mkdir -p "$TEST_IMAGE_DIR"
exec "$REMOTE_LAUNCHER" \
  --test-images "$TEST_IMAGE_DIR" \
  --location-iou-threshold 0.4 \
  --threshold 0.4 \
  --stable-wait-ms 500 \
  --blank-width 1200 \
  --blank-height 900 \
  --window-x 50 \
  --window-y 50 \
  "$@"
