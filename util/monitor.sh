#!/usr/bin/env bash
set -euo pipefail
#
# Accept any training log so monitoring is not tied to one experiment directory.
#
LOG_FILE="${1:-}"
TAIL_LINES="${2:-300}"
if [[ -z "$LOG_FILE" ]]; then
  echo "Usage: $0 <training-log> [initial-line-count]" >&2
  exit 2
fi
if [[ ! -f "$LOG_FILE" ]]; then
  echo "Training log not found: $LOG_FILE" >&2
  exit 1
fi
if [[ ! "$TAIL_LINES" =~ ^[0-9]+$ ]]; then
  echo "Initial line count must be a non-negative integer: $TAIL_LINES" >&2
  exit 2
fi
#
# Follow by filename so monitoring survives the log being replaced during a restart.
#
# Use POSIX match and substr operations for compatibility with Jetson awk variants.
#
tail -n "$TAIL_LINES" -F -- "$LOG_FILE" | \
  awk '
    /Epoch: \[[0-9]+\][[:space:]]+\[[[:space:]]*[0-9]+\/[0-9]+\]/ {
      match($0, /Epoch: \[[0-9]+\][[:space:]]+\[[[:space:]]*[0-9]+\/[0-9]+\]/)
      progress = substr($0, RSTART, RLENGTH)
      match($0, /eta: [0-9:]+/)
      eta = substr($0, RSTART, RLENGTH)
      match($0, /lr: [0-9.]+/)
      lr = substr($0, RSTART, RLENGTH)
      match($0, /class_error: [0-9.]+/)
      class_error = substr($0, RSTART, RLENGTH)
      match($0, /loss: [0-9.]+ \([0-9.]+\)/)
      loss = substr($0, RSTART, RLENGTH)
      print strftime("%Y-%m-%d %H:%M:%S"), "TRAIN_BATCH", progress, eta, lr, class_error, loss
      fflush()
    }
    /Epoch: \[[0-9]+\] Total time:/ {
      print strftime("%Y-%m-%d %H:%M:%S"), "TRAIN_TOTAL", $0
      fflush()
    }
    /^Test:[[:space:]]+\[[[:space:]]*[0-9]+\/[0-9]+\]/ {
      match($0, /^Test:[[:space:]]+\[[[:space:]]*[0-9]+\/[0-9]+\]/)
      progress = substr($0, RSTART, RLENGTH)
      match($0, /eta: [0-9:]+/)
      eta = substr($0, RSTART, RLENGTH)
      match($0, /class_error: [0-9.]+/)
      class_error = substr($0, RSTART, RLENGTH)
      match($0, /loss: [0-9.]+ \([0-9.]+\)/)
      loss = substr($0, RSTART, RLENGTH)
      print strftime("%Y-%m-%d %H:%M:%S"), "TEST_BATCH", progress, eta, class_error, loss
      fflush()
    }
    /^Test: Total time:/ {
      print strftime("%Y-%m-%d %H:%M:%S"), "TEST_TOTAL", $0
      fflush()
    }
    /^Averaged stats:/ {
      print strftime("%Y-%m-%d %H:%M:%S"), "AVG_STATS", $0
      fflush()
    }
    /^ Average (Precision|Recall)/ {
      print strftime("%Y-%m-%d %H:%M:%S"), "COCO_METRIC", $0
      fflush()
    }
  '
