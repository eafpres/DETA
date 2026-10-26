#!/usr/bin/env bash
set -euo pipefail
#
# Retain the original entry point while resolving it from the current checkout.
#
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_LAUNCHER="${DETA_INFERENCE_LAUNCHER:-$REPO_DIR/run_deta_infer.sh}"
if [[ ! -x "$INFERENCE_LAUNCHER" ]]; then
  echo "DETA inference launcher is missing or not executable: $INFERENCE_LAUNCHER" >&2
  exit 1
fi
exec "$INFERENCE_LAUNCHER" "$@"
