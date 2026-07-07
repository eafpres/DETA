#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:34:48 2026

@author: eafpres
"""

import argparse
import ast
import json
import re
from pathlib import Path
import pandas as pd
NUM_RE = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?"
EPOCH_RE = re.compile(r"\bEpoch:\s*\[(\d+)\]", re.IGNORECASE)
KV_RE = re.compile(rf"\b([A-Za-z_][A-Za-z0-9_]*)=({NUM_RE})")
AVG_LOSS_RE = re.compile(
  rf"\bloss:\s*(?P<loss>{NUM_RE})(?:\s*\((?P<loss_avg>{NUM_RE})\))?"
)
CLASS_ID_RE = re.compile(r"\b(?:class_id|category_id)=(-?\d+)")
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
def parse_mapping(line):
  text = line.strip()
  if not text.startswith("{") or not text.endswith("}"):
    return None
  try:
    value = json.loads(text)
    if isinstance(value, dict):
      return value
  except json.JSONDecodeError:
    pass
  try:
    value = ast.literal_eval(text)
    if isinstance(value, dict):
      return value
  except (SyntaxError, ValueError):
    pass
  return None
def parse_kv(line):
  values = {}
  for match in KV_RE.finditer(line):
    values[match.group(1)] = safe_float(match.group(2))
  return values
def update_epoch_row(rows_by_epoch, epoch, values):
  if epoch is None:
    return
  if epoch not in rows_by_epoch:
    rows_by_epoch[epoch] = {
      "epoch": epoch,
      "val_loss": None,
      "macro_f1": None,
      "micro_f1": None,
      "macro_precision": None,
      "macro_recall": None,
      "micro_precision": None,
      "micro_recall": None,
      "tp": None,
      "fp": None,
      "fn": None
    }
  for key, value in values.items():
    if value is not None:
      rows_by_epoch[epoch][key] = value
def parse_json_epoch_row(obj):
  epoch = safe_int(obj.get("epoch"))
  values = {
    "val_loss": safe_float(obj.get("test_loss", obj.get("val_loss"))),
    "macro_f1": safe_float(obj.get("test_val_f1_per_class_iou_0_50", obj.get("val_f1_per_class_iou_0_50"))),
    "macro_precision": safe_float(obj.get("test_val_f1_per_class_iou_0_50_precision", obj.get("val_f1_per_class_iou_0_50_precision"))),
    "macro_recall": safe_float(obj.get("test_val_f1_per_class_iou_0_50_recall", obj.get("val_f1_per_class_iou_0_50_recall")))
  }
  return epoch, values
def parse_val_f1_line(line, current_epoch):
  values = parse_kv(line)
  row = {
    "macro_precision": values.get("precision"),
    "macro_recall": values.get("recall"),
    "macro_f1": values.get("f1"),
    "tp": values.get("tp"),
    "fp": values.get("fp"),
    "fn": values.get("fn"),
    "micro_precision": values.get("micro_precision"),
    "micro_recall": values.get("micro_recall"),
    "micro_f1": values.get("micro_f1")
  }
  return current_epoch, row
def parse_validation_averaged_stats(line, current_epoch):
  match = AVG_LOSS_RE.search(line)
  if match is None:
    return current_epoch, {}
  loss_value = match.group("loss_avg")
  if loss_value is None:
    loss_value = match.group("loss")
  return current_epoch, {"val_loss": safe_float(loss_value)}
def update_class_row(class_rows, epoch, class_id, values):
  if epoch is None or class_id is None:
    return
  key = (epoch, str(class_id))
  if key not in class_rows:
    class_rows[key] = {
      "epoch": epoch,
      "class_name": str(class_id),
      "f1": None,
      "tp": None,
      "fp": None,
      "fn": None,
      "precision": None,
      "recall": None,
      "threshold": None,
      "mean_score_tp": None,
      "mean_score_fp": None
    }
  for metric, value in values.items():
    if value is not None:
      class_rows[key][metric] = value
def parse_class_line(line, current_epoch, class_rows):
  match = CLASS_ID_RE.search(line)
  if match is None:
    return
  class_id = match.group(1)
  values = parse_kv(line)
  update_class_row(
    class_rows,
    current_epoch,
    class_id,
    {
      "f1": values.get("f1"),
      "tp": values.get("tp"),
      "fp": values.get("fp"),
      "fn": values.get("fn"),
      "precision": values.get("precision"),
      "recall": values.get("recall"),
      "threshold": values.get("threshold"),
      "mean_score_tp": values.get("mean_score_tp"),
      "mean_score_fp": values.get("mean_score_fp")
    }
  )
def parse_json_class_metrics(obj, class_rows):
  epoch = safe_int(obj.get("epoch"))
  metrics = obj.get("test_val_f1_per_class_iou_0_50_class_metrics")
  if metrics is None:
    metrics = obj.get("val_f1_per_class_iou_0_50_class_metrics")
  if not isinstance(metrics, dict):
    return
  for class_id, class_metric in metrics.items():
    if not isinstance(class_metric, dict):
      continue
    update_class_row(
      class_rows,
      epoch,
      class_id,
      {
        "f1": safe_float(class_metric.get("f1")),
        "tp": safe_float(class_metric.get("tp")),
        "fp": safe_float(class_metric.get("fp")),
        "fn": safe_float(class_metric.get("fn")),
        "precision": safe_float(class_metric.get("precision")),
        "recall": safe_float(class_metric.get("recall")),
        "threshold": safe_float(class_metric.get("threshold")),
        "mean_score_tp": safe_float(class_metric.get("mean_score_tp")),
        "mean_score_fp": safe_float(class_metric.get("mean_score_fp"))
      }
    )
def parse_log(log_file):
  current_epoch = None
  current_phase = None
  rows_by_epoch = {}
  class_rows = {}
  with open(log_file, "r", encoding="utf-8", errors="replace") as file:
    for raw_line in file:
      line = raw_line.strip()
      if line == "":
        continue
      match = EPOCH_RE.search(line)
      if match is not None:
        current_epoch = safe_int(match.group(1))
        current_phase = "train"
      if line.startswith("Test:"):
        current_phase = "val"
      obj = parse_mapping(line)
      if obj is not None:
        epoch, values = parse_json_epoch_row(obj)
        if epoch is not None:
          current_epoch = epoch
        update_epoch_row(rows_by_epoch, current_epoch, values)
        parse_json_class_metrics(obj, class_rows)
        continue
      if line.startswith("Averaged stats:") and current_phase == "val":
        epoch, values = parse_validation_averaged_stats(line, current_epoch)
        update_epoch_row(rows_by_epoch, epoch, values)
        current_phase = None
        continue
      if line.startswith("VAL_F1_PER_CLASS"):
        epoch, values = parse_val_f1_line(line, current_epoch)
        update_epoch_row(rows_by_epoch, epoch, values)
        continue
      if line.startswith("VAL_F1_CLASS"):
        parse_class_line(line, current_epoch, class_rows)
  epoch_df = pd.DataFrame(list(rows_by_epoch.values()))
  if len(epoch_df) == 0:
    epoch_df = pd.DataFrame(columns=[
      "epoch",
      "val_loss",
      "macro_f1",
      "micro_f1",
      "macro_precision",
      "macro_recall",
      "micro_precision",
      "micro_recall",
      "tp",
      "fp",
      "fn"
    ])
  else:
    epoch_df = epoch_df.sort_values("epoch").reset_index(drop=True)
  class_df = pd.DataFrame(list(class_rows.values()))
  if len(class_df) == 0:
    class_df = pd.DataFrame(columns=[
      "epoch",
      "class_name",
      "f1",
      "tp",
      "fp",
      "fn",
      "precision",
      "recall",
      "threshold",
      "mean_score_tp",
      "mean_score_fp"
    ])
  else:
    class_df = class_df.sort_values(["epoch", "class_name"]).reset_index(drop=True)
  return epoch_df, class_df
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-file", required=True, type=Path)
  parser.add_argument("--out-dir", required=True, type=Path)
  args = parser.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  epoch_df, class_df = parse_log(args.log_file)
  epoch_path = args.out_dir / "epoch_metrics.csv"
  class_path = args.out_dir / "class_metrics.csv"
  epoch_df.to_csv(epoch_path, index=False)
  class_df.to_csv(class_path, index=False)
  print(f"wrote: {epoch_path}")
  print(f"wrote: {class_path}")
  print(f"epoch rows: {len(epoch_df):,}")
  print(f"class rows: {len(class_df):,}")
  if len(epoch_df) > 0:
    print(epoch_df.tail(10).to_string(index=False))
  if len(class_df) == 0:
    print("no explicit per-class metric rows found")
  elif class_df[["mean_score_tp", "mean_score_fp"]].notna().sum().sum() == 0:
    print("per-class rows found, but no TP/FP mean scores were present")
if __name__ == "__main__":
  main()
