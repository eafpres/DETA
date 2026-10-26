#!/usr/bin/env bash
set -euo pipefail
#
# Delegate to the reusable test wrapper so test and normal inference share setup.
#
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_LAUNCHER="${DETA_TEST_LAUNCHER:-$REPO_DIR/tools/run_deta_infer_test_jetson_remote.sh}"
if [[ ! -x "$TEST_LAUNCHER" ]]; then
  echo "Jetson test launcher is missing or not executable: $TEST_LAUNCHER" >&2
  exit 1
fi
exec "$TEST_LAUNCHER" "$@"
