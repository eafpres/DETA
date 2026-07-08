#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monitor DETA batch loss in a minimal frameless Qt window."""
import argparse
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.style
matplotlib.style.use("dark_background")
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure

TRAIN_RE = re.compile(
  r"Epoch: \[(?P<epoch>\d+)\]\s+"
  r"\[\s*(?P<batch>\d+)\/(?P<total>\d+)\].*?"
  r"loss: (?P<loss>[0-9.]+) \((?P<loss_avg>[0-9.]+)\)"
)

ETA_RE = re.compile(
  r"\beta:\s+(?P<eta>\d+:\d{2}:\d{2})"
)

VAL_RE = re.compile(
  r"(?:^|\s)Test:\s+"
  r"\[\s*(?P<batch>\d+)\/(?P<total>\d+)\].*?"
  r"loss: (?P<loss>[0-9.eE+-]+) \((?P<loss_avg>[0-9.eE+-]+)\)"
)

VAL_F1_RE = re.compile(
  r"VAL_F1\s+"
  r"iou=(?P<iou>[0-9.]+)\s+"
  r"threshold=(?P<threshold>[0-9.]+)\s+"
  r"precision=(?P<precision>[0-9.]+)\s+"
  r"recall=(?P<recall>[0-9.]+)\s+"
  r"f1=(?P<f1>[0-9.]+)\s+"
  r"tp=(?P<tp>\d+)\s+"
  r"fp=(?P<fp>\d+)\s+"
  r"fn=(?P<fn>\d+)"
)

VAL_F1_PER_CLASS_RE = re.compile(
  r"VAL_F1_PER_CLASS\s+"
  r"iou=(?P<iou>[0-9.]+)\s+"
  r"precision=(?P<precision>[0-9.]+)\s+"
  r"recall=(?P<recall>[0-9.]+)\s+"
  r"f1=(?P<f1>[0-9.]+)\s+"
  r"tp=(?P<tp>\d+)\s+"
  r"fp=(?P<fp>\d+)\s+"
  r"fn=(?P<fn>\d+)"
)

def qt_enum(name, nested_name):
  """Return a Qt enum value for either Qt 5 or Qt 6.
  Args:
    name: Enum member name.
    nested_name: Qt 6 nested enum class name.
  Returns:
    object: Requested Qt enum value.
  """
  value = getattr(QtCore.Qt, name, None)
  if value is not None:
    return value
  return getattr(getattr(QtCore.Qt, nested_name), name)
def qevent_enum(name):
  """Return a QEvent enum value for either Qt 5 or Qt 6.
  Args:
    name: Event enum member name.
  Returns:
    object: Requested QEvent enum value.
  """
  value = getattr(QtCore.QEvent, name, None)
  if value is not None:
    return value
  return getattr(QtCore.QEvent.Type, name)
FRAMELESS_HINT = qt_enum("FramelessWindowHint", "WindowType")
LEFT_EDGE = qt_enum("LeftEdge", "Edge")
RIGHT_EDGE = qt_enum("RightEdge", "Edge")
TOP_EDGE = qt_enum("TopEdge", "Edge")
BOTTOM_EDGE = qt_enum("BottomEdge", "Edge")
def no_edge_value():
  """Return an empty Qt edge-mask value for either Qt 5 or Qt 6.
  Returns:
    object: Empty Qt edge-mask value.
  """
  edges_type = getattr(QtCore.Qt, "Edges", None)
  if edges_type is not None:
    return edges_type(0)
  return QtCore.Qt.Edge(0)
NO_EDGE = no_edge_value()
LEFT_BUTTON = qt_enum("LeftButton", "MouseButton")
SIZE_H_CURSOR = qt_enum("SizeHorCursor", "CursorShape")
SIZE_V_CURSOR = qt_enum("SizeVerCursor", "CursorShape")
SIZE_F_CURSOR = qt_enum("SizeFDiagCursor", "CursorShape")
SIZE_B_CURSOR = qt_enum("SizeBDiagCursor", "CursorShape")
SIZE_ALL_CURSOR = qt_enum("SizeAllCursor", "CursorShape")
ARROW_CURSOR = qt_enum("ArrowCursor", "CursorShape")
MOUSE_PRESS = qevent_enum("MouseButtonPress")
MOUSE_RELEASE = qevent_enum("MouseButtonRelease")
MOUSE_MOVE = qevent_enum("MouseMove")
KEY_PRESS = qevent_enum("KeyPress")
def global_position(event):
  """Return an event global position for either Qt 5 or Qt 6.
  Args:
    event: Qt mouse event.
  Returns:
    QtCore.QPoint: Global mouse position.
  """
  if hasattr(event, "globalPosition"):
    return event.globalPosition().toPoint()
  return event.globalPos()
def local_position(event):
  """Return an event local position for either Qt 5 or Qt 6.
  Args:
    event: Qt mouse event.
  Returns:
    QtCore.QPoint: Local mouse position.
  """
  if hasattr(event, "position"):
    return event.position().toPoint()
  return event.pos()
def parse_args():
  """Parse command-line arguments.
  Returns:
    argparse.Namespace: Parsed command-line arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-file", required = True)
  parser.add_argument("--refresh-ms", type = int, default = 5000)
  parser.add_argument("--max-points", type = int, default = 500000)
  parser.add_argument("--border-width", type = int, default = 1)
  parser.add_argument("--border-color", default = "#555555")
  parser.add_argument("--resize-margin", type = int, default = 5)
  parser.add_argument("--drag-height", type = int, default = 38)
  parser.add_argument("--y-min", type = float, default = None)
  parser.add_argument("--y-max", type = float, default = None)
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
    eta_match = ETA_RE.search(line)
    record = {
      "epoch": epoch,
      "batch": int(train_match.group("batch")),
      "total": int(train_match.group("total")),
      "loss": float(train_match.group("loss")),
      "loss_avg": float(train_match.group("loss_avg")),
      "eta": (
        eta_match.group("eta")
        if eta_match is not None
        else None
      )
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

  val_f1_per_class_match = VAL_F1_PER_CLASS_RE.search(line)

  if val_f1_per_class_match is not None and current_epoch is not None:
    record = {
      "epoch": current_epoch,
      "mode": "per-class thresholds",
      "iou": float(val_f1_per_class_match.group("iou")),
      "threshold": None,
      "precision": float(val_f1_per_class_match.group("precision")),
      "recall": float(val_f1_per_class_match.group("recall")),
      "f1": float(val_f1_per_class_match.group("f1")),
      "tp": int(val_f1_per_class_match.group("tp")),
      "fp": int(val_f1_per_class_match.group("fp")),
      "fn": int(val_f1_per_class_match.group("fn"))
    }
    return "val_f1", record, current_epoch

  val_f1_match = VAL_F1_RE.search(line)
  if val_f1_match is not None and current_epoch is not None:
    record = {
      "epoch": current_epoch,
      "mode": "global threshold",
      "iou": float(val_f1_match.group("iou")),
      "threshold": float(val_f1_match.group("threshold")),
      "precision": float(val_f1_match.group("precision")),
      "recall": float(val_f1_match.group("recall")),
      "f1": float(val_f1_match.group("f1")),
      "tp": int(val_f1_match.group("tp")),
      "fp": int(val_f1_match.group("fp")),
      "fn": int(val_f1_match.group("fn"))
    }
    return "val_f1", record, current_epoch
  return None, None, current_epoch
def append_train_record(
  train_records,
  val_records,
  val_f1_records,
  record
):
  """Append a training record and remove stale data after a resume.
  Args:
    train_records: Previously parsed training records.
    val_records: Previously parsed validation records.
    val_f1_records: Previously parsed validation F1 records.
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
  val_f1_records = []
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
          val_f1_records = val_f1_records,
          record = record
        )
      elif record_type == "val":
        val_records.append(record)
      elif record_type == "val_f1":
        val_f1_records.append(record)
    offset = f.tell()
  return (
    train_records,
    val_records,
    val_f1_records,
    offset,
    current_epoch
  )
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
class LossMonitorWindow(QtWidgets.QMainWindow):
  """Display the loss plot in a minimal movable and resizable window."""
  def __init__(self, args, log_path):
    """Initialize the monitor window.
    Args:
      args: Parsed command-line arguments.
      log_path: Path to the DETA log file.
    """
    super().__init__()
    self.args = args
    self.log_path = log_path
    self.resize_margin = max(args.resize_margin, args.border_width)
    self.drag_height = args.drag_height
    self.resize_edges = NO_EDGE
    self.resize_start_global = None
    self.resize_start_geometry = None
    self.drag_offset = None
    (
      self.train_records,
      self.val_records,
      self.val_f1_records,
      self.offset,
      self.current_epoch
    ) = load_existing_lines(log_path)
    self.setWindowFlag(FRAMELESS_HINT, True)
    self.setWindowTitle("DETA loss monitor")
    self.setMinimumSize(520, 300)
    self.resize(750, 400)
    frame = QtWidgets.QWidget()
    frame.setObjectName("frame")
    frame.setStyleSheet(
      f"QWidget#frame {{ background-color: #111111; "
      f"border: {args.border_width}px solid {args.border_color}; }}"
    )
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(
      args.border_width,
      args.border_width,
      args.border_width,
      args.border_width
    )
    layout.setSpacing(0)
    self.figure = Figure(figsize = (9, 5))
    self.figure.patch.set_facecolor("#111111")
    self.canvas = FigureCanvas(self.figure)
    layout.addWidget(self.canvas, 1)
    self.setCentralWidget(frame)
    self.ax = self.figure.subplots()
    self.ax.set_facecolor("#111111")
    self.batch_line, = self.ax.plot([], [], label = "train batch loss")
    self.train_avg_line, = self.ax.plot(
      [],
      [],
      label = "train running average"
    )
    self.val_line, = self.ax.plot(
      [],
      [],
      marker = "o",
      linewidth = 2,
      label = "validation loss"
    )
    self.val_annotations = []
    self.val_f1_text = self.ax.text(
      0.015,
      0.975,
      "",
      transform = self.ax.transAxes,
      fontsize = 6,
      fontfamily = "monospace",
      verticalalignment = "top",
      horizontalalignment = "left",
      bbox = {
        "boxstyle": "round,pad=0.35",
        "facecolor": "#111111",
        "edgecolor": "#777777",
        "alpha": 0.6
      }
    )
    self.ax.set_xlabel("cumulative training batch")
    self.ax.set_ylabel("loss")
    self.ax.grid(visible = True, alpha = 0.25)
    self.ax.legend(fontsize = 8, title_fontsize = 10, loc = 'upper right')
    self.install_window_filter(frame)
    self.update_plot()
    self.timer = QtCore.QTimer(self)
    self.timer.timeout.connect(self.update_plot)
    self.timer.start(args.refresh_ms)
  def install_window_filter(self, widget):
    """Install move and resize handling on a widget and its children.
    Args:
      widget: Qt widget to configure recursively.
    """
    widget.setMouseTracking(True)
    widget.installEventFilter(self)
    for child in widget.findChildren(QtWidgets.QWidget):
      child.setMouseTracking(True)
      child.installEventFilter(self)
  def edges_at(self, point):
    """Return resize edges for a point in window coordinates.
    Args:
      point: Point relative to this window.
    Returns:
      object: Qt edge flags at the point.
    """
    edges = NO_EDGE
    if point.x() <= self.resize_margin:
      edges |= LEFT_EDGE
    elif point.x() >= self.width() - self.resize_margin:
      edges |= RIGHT_EDGE
    if point.y() <= self.resize_margin:
      edges |= TOP_EDGE
    elif point.y() >= self.height() - self.resize_margin:
      edges |= BOTTOM_EDGE
    return edges
  def is_drag_area(self, point):
    """Return whether a point is in the invisible plot drag band.
    Args:
      point: Point relative to this window.
    Returns:
      bool: Whether the point can initiate window movement.
    """
    return point.y() <= self.drag_height and self.edges_at(point) == NO_EDGE
  def cursor_for_point(self, point):
    """Return the appropriate cursor for a window point.
    Args:
      point: Point relative to this window.
    Returns:
      object: Qt cursor shape.
    """
    edges = self.edges_at(point)
    if (
      edges == (LEFT_EDGE | TOP_EDGE) or
      edges == (RIGHT_EDGE | BOTTOM_EDGE)
    ):
      return SIZE_F_CURSOR
    if (
      edges == (RIGHT_EDGE | TOP_EDGE) or
      edges == (LEFT_EDGE | BOTTOM_EDGE)
    ):
      return SIZE_B_CURSOR
    if edges & (LEFT_EDGE | RIGHT_EDGE):
      return SIZE_H_CURSOR
    if edges & (TOP_EDGE | BOTTOM_EDGE):
      return SIZE_V_CURSOR
    if self.is_drag_area(point):
      return SIZE_ALL_CURSOR
    return ARROW_CURSOR
  def eventFilter(self, watched, event):
    """Handle border resizing and movement from the plot title region.
    Args:
      watched: Widget that received the event.
      event: Qt event.
    Returns:
      bool: Whether the event has been consumed.
    """
    if isinstance(watched, QtWidgets.QWidget):
      if event.type() == MOUSE_MOVE:
        point = watched.mapTo(self, local_position(event))
        self.setCursor(self.cursor_for_point(point))
        if self.resize_edges != NO_EDGE and event.buttons() & LEFT_BUTTON:
          self.apply_fallback_resize(global_position(event))
          return True
        if self.drag_offset is not None and event.buttons() & LEFT_BUTTON:
          self.move(global_position(event) - self.drag_offset)
          return True
      elif event.type() == MOUSE_PRESS:
        if event.button() == LEFT_BUTTON and not self.isMaximized():
          point = watched.mapTo(self, local_position(event))
          edges = self.edges_at(point)
          if edges != NO_EDGE:
            window_handle = self.windowHandle()
            if window_handle is not None:
              if hasattr(window_handle, "startSystemResize"):
                if window_handle.startSystemResize(edges):
                  return True
            self.resize_edges = edges
            self.resize_start_global = global_position(event)
            self.resize_start_geometry = self.geometry()
            return True
          if self.is_drag_area(point):
            window_handle = self.windowHandle()
            if window_handle is not None:
              if hasattr(window_handle, "startSystemMove"):
                if window_handle.startSystemMove():
                  return True
            self.drag_offset = (
              global_position(event) - self.frameGeometry().topLeft()
            )
            return True
      elif event.type() == MOUSE_RELEASE:
        self.resize_edges = NO_EDGE
        self.resize_start_global = None
        self.resize_start_geometry = None
        self.drag_offset = None
    return super().eventFilter(watched, event)
  def apply_fallback_resize(self, current_global):
    """Resize manually when the compositor declines native resizing.
    Args:
      current_global: Current global cursor position.
    """
    if self.resize_start_global is None:
      return
    delta = current_global - self.resize_start_global
    geometry = QtCore.QRect(self.resize_start_geometry)
    if self.resize_edges & LEFT_EDGE:
      geometry.setLeft(geometry.left() + delta.x())
    if self.resize_edges & RIGHT_EDGE:
      geometry.setRight(geometry.right() + delta.x())
    if self.resize_edges & TOP_EDGE:
      geometry.setTop(geometry.top() + delta.y())
    if self.resize_edges & BOTTOM_EDGE:
      geometry.setBottom(geometry.bottom() + delta.y())
    if geometry.width() < self.minimumWidth():
      if self.resize_edges & LEFT_EDGE:
        geometry.setLeft(geometry.right() - self.minimumWidth())
      else:
        geometry.setRight(geometry.left() + self.minimumWidth())
    if geometry.height() < self.minimumHeight():
      if self.resize_edges & TOP_EDGE:
        geometry.setTop(geometry.bottom() - self.minimumHeight())
      else:
        geometry.setBottom(geometry.top() + self.minimumHeight())
    self.setGeometry(geometry)
  def read_appended_lines(self):
    """Read any newly appended log lines and update parsed records."""
    try:
      if self.log_path.stat().st_size < self.offset:
        (
          self.train_records,
          self.val_records,
          self.val_f1_records,
          self.offset,
          self.current_epoch
        ) = load_existing_lines(self.log_path)
        return
      with self.log_path.open(
        "r",
        encoding = "utf-8",
        errors = "replace"
      ) as f:
        f.seek(self.offset)
        for line in f:
          record_type, record, self.current_epoch = parse_log_line(
            line = line,
            current_epoch = self.current_epoch
          )
          if record_type == "train":
            append_train_record(
              train_records = self.train_records,
              val_records = self.val_records,
              val_f1_records = self.val_f1_records,
              record = record
            )
          elif record_type == "val":
            self.val_records.append(record)
          elif record_type == "val_f1":
            self.val_f1_records.append(record)
        self.offset = f.tell()
    except FileNotFoundError:
      return
  def update_plot(self):
    """Read appended lines and refresh the displayed plot."""
    self.read_appended_lines()
    if not self.train_records:
      return
    visible = self.train_records[-self.args.max_points:]
    x_values = [
      record["epoch"] * record["total"] + record["batch"]
      for record in visible
    ]
    loss_values = [record["loss"] for record in visible]
    avg_values = [record["loss_avg"] for record in visible]
    self.batch_line.set_data(x_values, loss_values)
    self.train_avg_line.set_data(x_values, avg_values)
    latest = self.train_records[-1]
    val_x, val_y = get_val_epoch_points(
      val_records = self.val_records,
      train_total = latest["total"]
    )
    self.val_line.set_data(val_x, val_y)
    for annotation in self.val_annotations:
      annotation.remove()
    self.val_annotations.clear()
    for x_value, y_value in zip(val_x, val_y):
      annotation = self.ax.annotate(
        text = f"{y_value:.3f}",
        xy = (x_value, y_value),
        xytext = (-6, 40),
        textcoords = "offset points",
        fontsize = 8,
        rotation = "vertical",
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
      self.val_annotations.append(annotation)
    self.ax.relim()
    self.ax.autoscale_view()
    self.ax.set_ylim(
      bottom = self.args.y_min,
      top = self.args.y_max
    )
    if self.val_f1_records:
      latest_val_f1 = self.val_f1_records[-1]
      threshold_text = (
        f"{latest_val_f1['threshold']:6.3f}"
        if latest_val_f1["threshold"] is not None
        else "per-class"
      )
      self.val_f1_text.set_text(
        f"validation using {latest_val_f1['mode']}\n"
        f"{'IoU':<21}{latest_val_f1['iou']:6.2f}\n"
        f"{'threshold':<21}{threshold_text}\n"
        f"{'macro precision':<21}{latest_val_f1['precision']:6.3f}\n"
        f"{'macro recall':<21}{latest_val_f1['recall']:6.3f}\n"
        f"{'macro f1':<21}{latest_val_f1['f1']:6.3f}\n"
        f"{'all-class tp/fp/fn':<21}"
        f"{latest_val_f1['tp']}/"
        f"{latest_val_f1['fp']}/"
        f"{latest_val_f1['fn']}"
      )
    else:
      self.val_f1_text.set_text("")
    eta_text = (
      f" | eta {latest['eta']}"
      if latest["eta"] is not None
      else ""
    )
    self.ax.set_title(
      f"DETA loss\n"
      f"epoch {latest['epoch']} batch "
      f"{latest['batch']}/{latest['total']} | "
      f"global batch {x_values[-1]}"
      f"{eta_text}"
    )
    self.canvas.draw_idle()
def main():
  """Run the Qt-based DETA loss monitor."""
  args = parse_args()
  log_path = Path(args.log_file).expanduser().resolve()
  if not log_path.is_file():
    raise FileNotFoundError(f"log file does not exist: {log_path}")
  app = QtWidgets.QApplication.instance()
  if app is None:
    app = QtWidgets.QApplication(sys.argv)
  window = LossMonitorWindow(args = args, log_path = log_path)
  window.show()
  exec_method = getattr(app, "exec", None)
  if exec_method is None:
    exec_method = app.exec_
  return exec_method()
if __name__ == "__main__":
  raise SystemExit(main())
