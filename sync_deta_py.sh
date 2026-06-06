#!/usr/bin/env bash
set -euo pipefail
SOURCE_DIR="/mnt/d/DETA/DETA/"
DEST_DIR="$HOME/DETA/DETA/"
#
# Confirm that both directories exist before copying files.
#
if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "ERROR: source directory does not exist: $SOURCE_DIR" >&2
  exit 1
fi
if [[ ! -d "$DEST_DIR/.git" ]]; then
  echo "ERROR: destination does not appear to be the WSL Git repo: $DEST_DIR" >&2
  exit 1
fi
#
# Copy only new or changed Python files while preserving subdirectories.
#
rsync \
  --archive \
  --verbose \
  --itemize-changes \
  --prune-empty-dirs \
  --include='*/' \
  --include='*.py' \
  --exclude='*' \
  "$SOURCE_DIR" \
  "$DEST_DIR"
echo
echo "Python-file sync complete."
echo
echo "Changed files in the WSL repo:"
git -C "$DEST_DIR" status --short
