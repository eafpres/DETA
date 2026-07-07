#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:04:44 2026

@author: eafpres
"""

import argparse
import re
from pathlib import Path
import pandas as pd
NUM_RE = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?"
EPOCH_RE = re.compile(r"\bEpoch:\s*\[(\d+)\]", re.IGNORECASE)
METER_RE = re.compile(
  rf"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
  rf"(?P<value>{NUM_RE})"
  rf"(?:\s*\((?P<avg>{NUM_RE})\))?"
)
def safe_float(value):
  try:
    if value is None:
      return None
    return float(value)
  except (TypeError, ValueError):
    return None
def safe_int(value):
  try:
    if value is None:
      return None
    return int(float(value))
  except (TypeError, ValueError):
    return None
def parse_averaged_stats(line):
  row = {}
  for match in METER_RE.finditer(line):
    key = match.group(1)
    value = match.group("avg")
    if value is None:
      value = match.group("value")
    value = safe_float(value)
    if value is not None:
      row[key] = value
  return row
def parse_log(log_file):
  current_epoch = None
  current_phase = None
  rows = []
  with open(log_file, "r", encoding="utf-8", errors="replace") as file:
    for raw_line in file:
      line = raw_line.strip()
      if line == "":
        continue
      epoch_match = EPOCH_RE.search(line)
      if epoch_match is not None:
        current_epoch = safe_int(epoch_match.group(1))
        current_phase = "train"
      if line.startswith("Test:"):
        current_phase = "val"
      if line.startswith("Averaged stats:") and current_phase == "train":
        row = parse_averaged_stats(line)
        if current_epoch is not None and row:
          row["epoch"] = current_epoch
          rows.append(row)
        current_phase = None
  if not rows:
    return pd.DataFrame()
  df = pd.DataFrame(rows)
  cols = ["epoch"] + [
    col
    for col in df.columns
    if col != "epoch"
  ]
  return df[cols].sort_values("epoch").reset_index(drop=True)
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-file", required=True, type=Path)
  parser.add_argument("--out-dir", required=True, type=Path)
  args = parser.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  train_df = parse_log(args.log_file)
  out_path = args.out_dir / "training_epoch_metrics.csv"
  train_df.to_csv(out_path, index=False)
  print(f"wrote: {out_path}")
  print(f"train epoch rows: {len(train_df):,}")
  if len(train_df) > 0:
    print(train_df.tail(10).to_string(index=False))
  else:
    print("no train averaged stats found")
if __name__ == "__main__":
  main()
