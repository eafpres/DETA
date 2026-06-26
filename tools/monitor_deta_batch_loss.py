#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:42:04 2026

@author: eafpres
"""
import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use("dark_background")
from matplotlib.animation import FuncAnimation
TRAIN_RE = re.compile(
  r"Epoch: \[(?P<epoch>\d+)\]\s+"
  r"\[\s*(?P<batch>\d+)\/(?P<total>\d+)\].*?"
  r"loss: (?P<loss>[0-9.]+) \((?P<loss_avg>[0-9.]+)\)"
)
VAL_RE = re.compile(
  r"^Test:\s+"
  r"\[\s*(?P<batch>\d+)\/(?P<total>\d+)\].*?"
  r"loss: (?P<loss>[0-9.]+) \((?P<loss_avg>[0-9.]+)\)"
)
def parse_args():
  """Parse command-line arguments.
  Returns:
    argparse.Namespace: Parsed command-line arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-file", required = True)
  parser.add_argument("--refresh-ms", type = int, default = 5000)
  parser.add_argument("--max-points", type = int, default = 500000)
  return parser.parse_args()
def parse_log_line(line, current_epoch):
  """Parse one DETA log line.
  Args:
    line: Raw log line.
    current_epoch: Most recently observed training epoch.
  Returns:
    tuple[str | None, dict | None, int]: Record type, record, epoch.
  """
  train_match = TRAIN_RE.search(line)
  if train_match is not None:
    epoch = int(train_match.group("epoch"))
    record = {
      "epoch": epoch,
      "batch": int(train_match.group("batch")),
      "total": int(train_match.group("total")),
      "loss": float(train_match.group("loss")),
      "loss_avg": float(train_match.group("loss_avg"))
    }
    return "train", record, epoch
  val_match = VAL_RE.search(line)
  if val_match is not None and current_epoch is not None:
    record = {
      "epoch": current_epoch,
      "batch": int(val_match.group("batch")),
      "total": int(val_match.group("total")),
      "loss": float(val_match.group("loss")),
      "loss_avg": float(val_match.group("loss_avg"))
    }
    return "val", record, current_epoch
  return None, None, current_epoch
def append_train_record(train_records, val_records, record):
  """Append a training record, removing stale data after a resume.
  Args:
    train_records: Previously parsed training records.
    val_records: Previously parsed validation records.
    record: Newly parsed training record.
  """
  if train_records:
    latest = train_records[-1]
    is_restart = (
      record["epoch"] < latest["epoch"] or
      (
        record["epoch"] == latest["epoch"] and
        record["batch"] <= latest["batch"]
      )
    )
    if is_restart:
      restart_epoch = record["epoch"]
      train_records[:] = [
        existing_record
        for existing_record in train_records
        if existing_record["epoch"] < restart_epoch
      ]
      val_records[:] = [
        existing_record
        for existing_record in val_records
        if existing_record["epoch"] < restart_epoch
      ]
  train_records.append(record)
def load_existing_lines(log_path):
  """Load existing train and validation records from a log file.
  Args:
    log_path: Log-file path.
  Returns:
    tuple[list[dict], list[dict], int, int | None]: Parsed records, offset,
    and current epoch.
  """
  train_records = []
  val_records = []
  current_epoch = None
  with log_path.open("r", encoding = "utf-8", errors = "replace") as f:
    for line in f:
      record_type, record, current_epoch = parse_log_line(
        line = line,
        current_epoch = current_epoch
      )
      if record_type == "train":
        append_train_record(
          train_records = train_records,
          val_records = val_records,
          record = record
        )
      elif record_type == "val":
        val_records.append(record)
    offset = f.tell()
  return train_records, val_records, offset, current_epoch
def get_val_epoch_points(val_records, train_total):
  """Return one validation-loss point per completed validation pass.
  Args:
    val_records: Validation batch records.
    train_total: Number of training batches per epoch.
  Returns:
    tuple[list[int], list[float]]: Cumulative-batch positions and losses.
  """
  latest_by_epoch = {}
  for record in val_records:
    latest_by_epoch[record["epoch"]] = record
  x_values = []
  y_values = []
  for epoch in sorted(latest_by_epoch):
    record = latest_by_epoch[epoch]
    x_values.append((epoch + 1) * train_total)
    y_values.append(record["loss_avg"])
  return x_values, y_values
def main():
  """Plot DETA training loss and validation loss as the log grows."""
  args = parse_args()
  log_path = Path(args.log_file).expanduser().resolve()
  if not log_path.is_file():
    raise FileNotFoundError(f"log file does not exist: {log_path}")
  train_records, val_records, offset, current_epoch = load_existing_lines(
    log_path
  )
  fig, ax = plt.subplots(
  figsize = (9, 5)
  )
  manager = plt.get_current_fig_manager()
  if hasattr(manager, "toolbar") and manager.toolbar is not None:
    manager.toolbar.pack_forget()
  if hasattr(manager, "window"):
    manager.window.overrideredirect(True)
  fig.patch.set_facecolor("#111111")
  ax.set_facecolor("#111111")
  batch_line, = ax.plot([], [], label = "train batch loss")
  train_avg_line, = ax.plot([], [], label = "train running average")
  val_line, = ax.plot(
    [],
    [],
    marker = "o",
    linewidth = 2,
    label = "validation loss"
  )
  val_annotations = []
  val_label_lines = []
  ax.set_xlabel("cumulative training batch")
  ax.set_ylabel("loss")
  ax.grid(
    visible = True,
    alpha = 0.25
  )
  ax.legend()
  def update(_):
    """Read appended log lines and update the plot.
    Args:
      _: Matplotlib animation frame.
    Returns:
      tuple: Updated Matplotlib objects.
    """
    nonlocal offset
    nonlocal current_epoch
    try:
      with log_path.open("r", encoding = "utf-8", errors = "replace") as f:
        f.seek(offset)
        for line in f:
          record_type, record, current_epoch = parse_log_line(
            line = line,
            current_epoch = current_epoch
          )
          if record_type == "train":
            append_train_record(
              train_records = train_records,
              val_records = val_records,
              record = record
            )
          elif record_type == "val":
            val_records.append(record)
        offset = f.tell()
    except FileNotFoundError:
      return batch_line, train_avg_line, val_line
    if not train_records:
      return batch_line, train_avg_line, val_line
    visible = train_records[-args.max_points:]
    x_values = [
      record["epoch"] * record["total"] + record["batch"]
      for record in visible
    ]
    loss_values = [record["loss"] for record in visible]
    avg_values = [record["loss_avg"] for record in visible]
    batch_line.set_data(x_values, loss_values)
    train_avg_line.set_data(x_values, avg_values)
    latest = train_records[-1]
    val_x, val_y = get_val_epoch_points(
      val_records = val_records,
      train_total = latest["total"]
    )
    val_line.set_data(val_x, val_y)
    for annotation in val_annotations:
      annotation.remove()
    val_annotations.clear()
    for x_value, y_value in zip(val_x, val_y):
      annotation = ax.annotate(
        text = f"{y_value:.3f}",
        xy = (x_value, y_value),
        xytext = (-6, 40),
        textcoords = "offset points",
        fontsize = 8,
        rotation = 'vertical',
        horizontalalignment = "center",
        verticalalignment = "bottom",
        arrowprops = {
          "arrowstyle": "-",
          "linewidth": 1.25,
          "alpha": 1.0,
          "color": "red",
          "shrinkA": 0,
          "shrinkB": 0
        }
      )
      val_annotations.append(annotation)
    ax.relim()
    ax.autoscale_view()
    ax.set_title(
      f"DETA loss\n"
      f"epoch {latest['epoch']} batch "
      f"{latest['batch']}/{latest['total']} | "
      f"global batch {x_values[-1]}"
    )
    return batch_line, train_avg_line, val_line
  anim = FuncAnimation(
    fig,
    update,
    interval = args.refresh_ms,
    cache_frame_data = False
  )
  plt.show()
  _ = anim
if __name__ == "__main__":
  main()
