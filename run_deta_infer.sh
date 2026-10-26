#!/usr/bin/env bash
set -euo pipefail
#
# Resolve the Jetson wrapper from this checkout so the repository can live anywhere.
#
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_LAUNCHER="${DETA_REMOTE_LAUNCHER:-$REPO_DIR/tools/run_deta_infer_jetson_remote.sh}"
#
# Keep the historical display defaults here while the shared wrapper owns model paths.
#
if [[ ! -x "$REMOTE_LAUNCHER" ]]; then
  echo "Jetson inference launcher is missing or not executable: $REMOTE_LAUNCHER" >&2
  exit 1
fi
exec "$REMOTE_LAUNCHER" \
  --location-iou-threshold 0.4 \
  --threshold 0.4 \
  --stable-wait-ms 500 \
  --blank-width 1200 \
  --blank-height 900 \
  --window-x 50 \
  --window-y 50 \
  "$@"
