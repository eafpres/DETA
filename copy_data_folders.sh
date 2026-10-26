#!/usr/bin/env bash
set -euo pipefail
#
# Require the remote endpoint instead of committing a user, address, or home path.
#
REMOTE_TARGET="${1:-${DETA_REMOTE_TARGET:-}}"
REMOTE_DIRECTORY="${2:-${DETA_REMOTE_DIRECTORY:-}}"
SOURCE_ROOT="${3:-${DETA_COPY_SOURCE_ROOT:-$PWD}}"
if [[ -z "$REMOTE_TARGET" || -z "$REMOTE_DIRECTORY" ]]; then
  echo "Usage: $0 <user@host> <remote-directory> [source-root]" >&2
  echo "The DETA_REMOTE_TARGET and DETA_REMOTE_DIRECTORY variables may supply the first two values." >&2
  exit 2
fi
#
# Validate every source before copying so a partial deployment cannot look successful.
#
SOURCE_NAMES=(inference_results test_images inference_images)
SOURCE_PATHS=()
for source_name in "${SOURCE_NAMES[@]}"; do
  source_path="$SOURCE_ROOT/$source_name"
  if [[ ! -d "$source_path" ]]; then
    echo "Source directory not found: $source_path" >&2
    exit 1
  fi
  SOURCE_PATHS+=("$source_path")
done
#
# Quote the remote mkdir argument for its shell while leaving scp to manage transfer.
#
printf -v REMOTE_DIRECTORY_QUOTED '%q' "$REMOTE_DIRECTORY"
ssh "$REMOTE_TARGET" "mkdir -p -- $REMOTE_DIRECTORY_QUOTED"
scp -r -- "${SOURCE_PATHS[@]}" "$REMOTE_TARGET:$REMOTE_DIRECTORY/"
