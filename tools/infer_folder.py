#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:46:47 2026

@author: eafpres
"""
#%% libraries
#
import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
from main import get_args_parser
from models import build_model
import traceback
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import (
  ssdlite320_mobilenet_v3_large as build_ssd_model
)
#
#%% configure
#
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
#
# display label mapping
#
DISPLAY_CLASS_NAMES = {
  "001_Porosity": "Porosity",
  "003_Burn_through": "Burn Through",
  "004_Uneven": "Uneven",
}
#
#%% function defs
#
# display label helper
#
def display_label_name(raw_name):
  """Return a human-readable display label.

  Args:
    raw_name: Raw class name from the dataset.

  Returns:
    str: Display label.
  """
  return DISPLAY_CLASS_NAMES.get(raw_name, raw_name)

def resolve_path(path, must_exist = False):
  """Resolve a path relative to the DETA repo root.
  Args:
    path: Input path string.
    must_exist: Whether the path must already exist.
  Returns:
    pathlib.Path | None: Resolved path.
  """
  if path is None:
    return None
  path = Path(path).expanduser()
  if not path.is_absolute():
    path = ROOT / path
  path = path.resolve()
  if must_exist and not path.exists():
    raise FileNotFoundError(f"path does not exist: {path}")
  return path

def make_output_path(output_dir, image_path):
  """Create an output PNG path for an annotated image.
  Args:
    output_dir: Output folder.
    image_path: Source image path.
  Returns:
    pathlib.Path: Output PNG path.
  """
  output_dir.mkdir(parents = True, exist_ok = True)
  return output_dir / f"{image_path.stem}_annotated.png"

def prune_old_annotated_images(output_dir, max_images = 4):
  """Delete older annotated PNG files, keeping only newest files.
  Args:
    output_dir: Output folder.
    max_images: Maximum number of PNG files to keep.
  """
  output_dir.mkdir(parents = True, exist_ok = True)
  paths = sorted(
    [
      path
      for path in output_dir.glob("*.png")
      if path.is_file()
    ],
    key = lambda path: path.stat().st_mtime,
    reverse = True
  )
  for path in paths[max_images:]:
    path.unlink()
    print(f"deleted old annotated image: {path}", flush = True)

def save_annotated_image(output_dir, image_path, annotated, max_images = 4):
  """Save an annotated image as PNG and keep only newest outputs.
  Args:
    output_dir: Output folder or None.
    image_path: Source image path.
    annotated: Annotated BGR image.
    max_images: Maximum number of annotated PNG files to keep.
  """
  if output_dir is None:
    return
  output_path = make_output_path(
    output_dir = output_dir,
    image_path = image_path
  )
  ok = cv2.imwrite(str(output_path), annotated)
  if not ok:
    raise RuntimeError(f"failed to write annotated image: {output_path}")
  prune_old_annotated_images(
    output_dir = output_dir,
    max_images = max_images
  )
  print(f"saved annotated image: {output_path}", flush = True)

def parse_args():
  """Parse command-line arguments.
  Returns:
    argparse.Namespace: Parsed command-line arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required = True)
  parser.add_argument("--image-dir", required = True)
  parser.add_argument("--output-dir", default = None)
  parser.add_argument("--classes", default = None)
  parser.add_argument("--threshold", type = float, default = 0.40)
  parser.add_argument("--image-size", type = int, default = 800)
  parser.add_argument("--blank-width", type = int, default = 900)
  parser.add_argument("--blank-height", type = int, default = 700)
  parser.add_argument("--window-x", type = int, default = 100)
  parser.add_argument("--window-y", type = int, default = 100)
  parser.add_argument("--poll-ms", type = int, default = 100)
  parser.add_argument("--stable-wait-ms", type = int, default = 200)
  parser.add_argument("--test-images", default = None)
  parser.add_argument("--score-mode", choices = ["sigmoid",
                                                 "softmax_no_object"],
                      default = "sigmoid"
                      )
  parser.add_argument("--location-iou-threshold", type = float, default = 0.4)
  parser.add_argument("--ssd-checkpoint", default = None)
  parser.add_argument("--ssd-threshold", type = float, default = 0.30)
  parser.add_argument("--ssd-location-iou-threshold", type = float,
                      default = 0.3)
  parser.add_argument("--ssd-deta-iou-threshold", type = float,
                      default = 0.3)
  return parser.parse_args()

def load_classes(classes_path):
  """Load class names from a text file.
  Args:
    classes_path: Path to class-name text file.
  Returns:
    list[str] | None: Class names or None.
  """
  if classes_path is None:
    return None
  classes_path = resolve_path(classes_path, must_exist = True)
  with classes_path.open("r", encoding = "utf-8") as f:
    return [line.strip() for line in f if line.strip()]

def clear_image_dir(image_dir):
  """Delete image and partial-copy files from the inference folder.
  Args:
    image_dir: Folder to clear.
  """
  image_dir.mkdir(parents = True, exist_ok = True)
  for path in image_dir.iterdir():
    if not path.is_file():
      continue
    if path.suffix.lower() in IMAGE_EXTS or path.name.endswith(".copying"):
      path.unlink()

def get_image_paths(image_dir):
  """Get image files in the inference folder.
  Args:
    image_dir: Folder to scan.
  Returns:
    list[pathlib.Path]: Sorted image paths.
  """
  return sorted(
    path
    for path in image_dir.iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS
  )

def wait_until_file_stable(path, stable_wait_ms):
  """Wait until a file is stable and readable as an image.
  Args:
    path: Image path.
    stable_wait_ms: Wait time between checks.
  Returns:
    bool: True if stable and readable, otherwise False.
  """
  last_size = -1
  stable_count = 0
  for _ in range(20):
    try:
      size = path.stat().st_size
      if size > 0 and size == last_size:
        stable_count += 1
      else:
        stable_count = 0
      last_size = size
      if stable_count >= 3:
        try:
          with Image.open(path) as img:
            img.verify()
          return True
        except Exception:
          pass
      time.sleep(stable_wait_ms / 1000.0)
    except FileNotFoundError:
      return False
    except PermissionError:
      time.sleep(stable_wait_ms / 1000.0)
  return False

def strip_module_prefix(state_dict):
  """Remove DataParallel module prefixes from a state dict.
  Args:
    state_dict: Model state dictionary.
  Returns:
    dict: Cleaned model state dictionary.
  """
  clean = {}
  for key, value in state_dict.items():
    if key.startswith("module."):
      clean[key[7:]] = value
    else:
      clean[key] = value
  return clean

def build_deta_from_checkpoint(checkpoint_path, device):
  """Build a DETA model and load a checkpoint.
  Args:
    checkpoint_path: Path to fine-tuned checkpoint.
    device: Torch device string.
  Returns:
    torch.nn.Module: Loaded model.
  """
  checkpoint = torch.load(
    checkpoint_path,
    map_location = "cpu",
    weights_only = False
  )
  deta_parser = argparse.ArgumentParser(
    parents = [get_args_parser()],
    add_help = False
  )
  model_args = deta_parser.parse_args([])
  if "args" in checkpoint:
    saved_args = checkpoint["args"]
    for key, value in vars(saved_args).items():
      setattr(model_args, key, value)
  model_args.device = device
  model, _, _ = build_model(model_args)
  if "model" in checkpoint:
    state_dict = checkpoint["model"]
  elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
  else:
    state_dict = checkpoint
  state_dict = strip_module_prefix(state_dict)
  load_result = model.load_state_dict(state_dict, strict = False)
  print(f"missing keys: {len(load_result.missing_keys)}")
  print(f"unexpected keys: {len(load_result.unexpected_keys)}")
  if len(load_result.missing_keys) > 0:
    print("first missing keys:", load_result.missing_keys[:10])
  if len(load_result.unexpected_keys) > 0:
    print("first unexpected keys:", load_result.unexpected_keys[:10])
  model.to(device)
  model.eval()
  return model

def make_transform(image_size):
  """Create image preprocessing transform.
  Args:
    image_size: Short-side resize size.
  Returns:
    torchvision.transforms.Compose: Transform pipeline.
  """
  return T.Compose([
    T.Resize(image_size),
    T.ToTensor(),
    T.Normalize(
      mean = [0.485, 0.456, 0.406],
      std = [0.229, 0.224, 0.225]
    )
  ])

def cxcywh_to_xyxy(boxes):
  """Convert boxes from cxcywh to xyxy.
  Args:
    boxes: Boxes in cxcywh format.
  Returns:
    torch.Tensor: Boxes in xyxy format.
  """
  cx, cy, w, h = boxes.unbind(-1)
  x0 = cx - 0.5 * w
  y0 = cy - 0.5 * h
  x1 = cx + 0.5 * w
  y1 = cy + 0.5 * h
  return torch.stack([x0, y0, x1, y1], dim = -1)

def predict_one(
  model,
  pil_image,
  transform,
  threshold,
  device,
  score_mode,
  class_names,
  location_iou_threshold
):
  """Run inference for one image.
  Args:
    model: Loaded DETA model.
    pil_image: PIL RGB image.
    transform: Image preprocessing transform.
    threshold: Minimum score threshold.
    device: Torch device string.
    score_mode: Score mode.
  Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: Boxes, scores, labels.
  """
  width, height = pil_image.size
  image_tensor = transform(pil_image).unsqueeze(0).to(device)
  with torch.inference_mode():
    outputs = model(image_tensor)
  logits = outputs["pred_logits"][0].detach().cpu()
  pred_boxes = outputs["pred_boxes"][0].detach().cpu()
  if score_mode == "softmax_no_object":
    probs = logits.softmax(-1)
    scores, labels = probs[:, :-1].max(-1)
  else:
    probs = logits.sigmoid()
    scores, labels = probs.max(-1)
  keep = scores >= threshold
  keep = scores >= threshold
  scores = scores[keep]
  labels = labels[keep]
  boxes = pred_boxes[keep]
  boxes = cxcywh_to_xyxy(boxes)
  scale = torch.tensor([width, height, width, height], dtype = torch.float32)
  boxes = boxes * scale
  boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width - 1)
  boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height - 1)
  scores, labels, boxes = filter_good_weld_labels(
    scores = scores,
    labels = labels,
    boxes = boxes,
    class_names = class_names
  )
  boxes, scores, labels = keep_best_per_location(
    boxes = boxes,
    scores = scores,
    labels = labels,
    iou_threshold = location_iou_threshold
  )
  return boxes.numpy(), scores.numpy(), labels.numpy()

def label_for_id(label_id, class_names):
  """Map class id to display label.

  Args:
    label_id: Integer class id.
    class_names: Optional class-name list.

  Returns:
    str: Display label.
  """
  label_id = int(label_id)
  if label_id == -1:
    return "Good Weld"
  if class_names is None:
    return f"class_{label_id}_no_classes"
  class_name_by_id = {}
  for raw_name in class_names:
    try:
      raw_id = int(raw_name.split("_", 1)[0])
    except ValueError:
      continue
    class_name_by_id[raw_id] = raw_name
  if label_id in class_name_by_id:
    return display_label_name(class_name_by_id[label_id])
  return f"class_{label_id}_out_of_range"

def draw_predictions(image_bgr, boxes, scores, labels, class_names):
  """Draw predictions onto an image.
  Args:
    image_bgr: OpenCV BGR image.
    boxes: Boxes in xyxy format.
    scores: Detection scores.
    labels: Class ids.
    class_names: Optional class-name list.
  Returns:
    np.ndarray: Annotated image.
  """
  out = image_bgr.copy()
  for box, score, label_id in zip(boxes, scores, labels):
    x0, y0, x1, y1 = box.astype(int).tolist()
    text = f"{label_for_id(label_id, class_names)} {score:.2f}"
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
    text_size, baseline = cv2.getTextSize(
      text,
      cv2.FONT_HERSHEY_SIMPLEX,
      1.5,
      1
    )
    tx0 = x0
    ty0 = max(0, y0 - text_size[1] - baseline - 4)
    tx1 = min(out.shape[1] - 1, x0 + text_size[0] + 4)
    ty1 = y0
    cv2.rectangle(out, (tx0, ty0), (tx1, ty1), (0, 255, 0), -1)
    cv2.putText(
      out,
      text,
      (x0 + 2, max(text_size[1] + 2, y0 - baseline - 2)),
      cv2.FONT_HERSHEY_SIMPLEX,
      1.0,
      (0, 0, 0),
      2,
      cv2.LINE_AA
    )
  return out

def make_blank_image(width, height):
  """Create a blank display image.
  Args:
    width: Image width.
    height: Image height.
  Returns:
    np.ndarray: Black BGR image.
  """
  return np.zeros((height, width, 3), dtype = np.uint8)

def show_blank(window_name, blank, poll_ms):
  """Show a blank image and poll for window events.
  Args:
    window_name: OpenCV window name.
    blank: Blank image.
    poll_ms: Wait time in milliseconds.
  """
  cv2.imshow(window_name, blank)
  cv2.waitKey(poll_ms)

def wait_for_key_or_new_image(window_name, image_dir, poll_ms):
  """Wait until a keypress or a new image arrives.
  Args:
    window_name: OpenCV window name.
    image_dir: Folder to watch for new images.
    poll_ms: Polling wait time in milliseconds.
  Returns:
    str: "key" if a key was pressed, otherwise "new_image".
  """
  while True:
    paths = get_image_paths(image_dir)
    if paths:
      print(
        f"new image detected while waiting: {paths[0].name}",
        flush = True
      )
      return "new_image"
    key = cv2.waitKey(poll_ms)
    if key not in (-1, 255):
      print(f"keypress detected: {key}", flush = True)
      return "key"

def get_test_image_paths(test_images_dir):
  """Get test images from the source test-image folder.
  Args:
    test_images_dir: Source folder with test images.
  Returns:
    list[pathlib.Path]: Sorted test-image paths.
  """
  if test_images_dir is None:
    return []
  return sorted(
    path
    for path in test_images_dir.iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS
  )

def delete_image(path, retries = 10, delay_sec = 0.25):
  """Delete one image path with retries.
  Args:
    path: Image path.
    retries: Number of delete attempts.
    delay_sec: Delay between attempts.
  Returns:
    bool: True if deleted or already missing.
  """
  for _ in range(retries):
    try:
      path.unlink()
      return True
    except FileNotFoundError:
      return True
    except PermissionError:
      time.sleep(delay_sec)
  print(f"could not delete locked file: {path}")
  return False

def process_image(
  path,
  model,
  ssd_model,
  transform,
  class_names,
  args,
  device
):
  """Run inference and return an annotated image.

  Args:
    path: Image path.
    model: Loaded DETA model.
    ssd_model: Optional loaded SSDlite model.
    transform: Image transform.
    class_names: Optional class names.
    args: Parsed command-line arguments.
    device: Torch device string.

  Returns:
    np.ndarray: Annotated BGR image.
  """
  pil_image = Image.open(path).convert("RGB")
  image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

  deta_boxes, deta_scores, deta_labels = predict_one(
    model = model,
    pil_image = pil_image,
    transform = transform,
    threshold = args.threshold,
    device = device,
    score_mode = args.score_mode,
    class_names = class_names,
    location_iou_threshold = args.location_iou_threshold
  )

  ssd_boxes = np.zeros((0, 4), dtype = np.float32)
  ssd_scores = np.zeros((0,), dtype = np.float32)
  ssd_labels = np.zeros((0,), dtype = np.int64)

  if ssd_model is not None:
    ssd_boxes, ssd_scores, ssd_labels = predict_good_weld_one(
      ssd_model = ssd_model,
      pil_image = pil_image,
      threshold = args.ssd_threshold,
      device = device,
      location_iou_threshold = args.ssd_location_iou_threshold,
    )
    ssd_boxes, ssd_scores, ssd_labels = suppress_ssd_by_deta_overlap(
      deta_boxes = deta_boxes,
      ssd_boxes = ssd_boxes,
      ssd_scores = ssd_scores,
      ssd_labels = ssd_labels,
      iou_threshold = args.ssd_deta_iou_threshold,
    )

  boxes, scores, labels = merge_deta_and_ssd(
    deta_boxes = deta_boxes,
    deta_scores = deta_scores,
    deta_labels = deta_labels,
    ssd_boxes = ssd_boxes,
    ssd_scores = ssd_scores,
    ssd_labels = ssd_labels,
  )

  annotated = draw_predictions(
    image_bgr = image_bgr,
    boxes = boxes,
    scores = scores,
    labels = labels,
    class_names = class_names
  )
  print(
    f"{path.name}: "
    f"deta={len(deta_scores)} "
    f"ssd_kept={len(ssd_scores)} "
    f"total={len(scores)}"
  )
  return annotated

def filter_good_weld_labels(scores, labels, boxes, class_names):
  """Remove good-weld detections from outputs.

  Args:
    scores: Detection scores.
    labels: Detection class ids.
    boxes: Detection boxes.
    class_names: Optional class-name list.

  Returns:
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Filtered outputs.
  """
  if class_names is None:
    return scores, labels, boxes
  class_name_by_id = {}
  for raw_name in class_names:
    try:
      raw_id = int(raw_name.split("_", 1)[0])
    except ValueError:
      continue
    class_name_by_id[raw_id] = raw_name
  keep_items = []
  for label in labels:
    label_id = int(label.item())
    raw_name = class_name_by_id.get(label_id, "")
    name = raw_name.lower()
    keep_items.append("good_weld" not in name and "good weld" not in name)
  if not keep_items:
    return scores, labels, boxes
  keep = torch.tensor(keep_items, dtype = torch.bool)
  return scores[keep], labels[keep], boxes[keep]

def box_iou_one_to_many(box, boxes):
  """Compute IoU from one box to many boxes.
  Args:
    box: Single xyxy box.
    boxes: Many xyxy boxes.
  Returns:
    torch.Tensor: IoU values.
  """
  x0 = torch.maximum(box[0], boxes[:, 0])
  y0 = torch.maximum(box[1], boxes[:, 1])
  x1 = torch.minimum(box[2], boxes[:, 2])
  y1 = torch.minimum(box[3], boxes[:, 3])
  inter_w = torch.clamp(x1 - x0, min = 0)
  inter_h = torch.clamp(y1 - y0, min = 0)
  inter = inter_w * inter_h
  box_area = torch.clamp(box[2] - box[0], min = 0) * torch.clamp(
    box[3] - box[1],
    min = 0
  )
  boxes_area = torch.clamp(boxes[:, 2] - boxes[:, 0], min = 0) * torch.clamp(
    boxes[:, 3] - boxes[:, 1],
    min = 0
  )
  union = box_area + boxes_area - inter
  return inter / torch.clamp(union, min = 1e-6)

def keep_best_per_location(boxes, scores, labels, iou_threshold = 0.4):
  """Keep the highest-scoring detection per overlapping location.
  Args:
    boxes: Detection boxes in xyxy format.
    scores: Detection scores.
    labels: Detection class ids.
    iou_threshold: IoU threshold for grouping boxes.
  Returns:
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Filtered detections.
  """
  if len(scores) <= 1:
    return boxes, scores, labels
  order = torch.argsort(scores, descending = True)
  keep_indices = []
  while len(order) > 0:
    best_idx = order[0]
    keep_indices.append(best_idx)
    if len(order) == 1:
      break
    remaining = order[1:]
    ious = box_iou_one_to_many(boxes[best_idx], boxes[remaining])
    order = remaining[ious < iou_threshold]
  keep_indices = torch.stack(keep_indices)
  return boxes[keep_indices], scores[keep_indices], labels[keep_indices]

def suppress_ssd_by_deta_overlap(
  deta_boxes,
  ssd_boxes,
  ssd_scores,
  ssd_labels,
  iou_threshold = 0.3,
):
  """Drop SSD boxes that overlap any DETA box.

  Args:
    deta_boxes: DETA boxes as numpy array [N, 4].
    ssd_boxes: SSD boxes as numpy array [M, 4].
    ssd_scores: SSD scores as numpy array [M].
    ssd_labels: SSD labels as numpy array [M].
    iou_threshold: IoU threshold for suppressing SSD by DETA.

  Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered SSD outputs.
  """
  if len(ssd_boxes) == 0 or len(deta_boxes) == 0:
    return ssd_boxes, ssd_scores, ssd_labels
  deta_boxes_t = torch.as_tensor(deta_boxes, dtype = torch.float32)
  ssd_boxes_t = torch.as_tensor(ssd_boxes, dtype = torch.float32)
  keep = []
  for ssd_box in ssd_boxes_t:
    ious = box_iou_one_to_many(ssd_box, deta_boxes_t)
    keep.append(bool(torch.max(ious).item() < iou_threshold))
  keep = np.array(keep, dtype = bool)
  return ssd_boxes[keep], ssd_scores[keep], ssd_labels[keep]

def merge_deta_and_ssd(
  deta_boxes,
  deta_scores,
  deta_labels,
  ssd_boxes,
  ssd_scores,
  ssd_labels,
):
  """Concatenate DETA and SSD detections.

  Args:
    deta_boxes: DETA boxes.
    deta_scores: DETA scores.
    deta_labels: DETA labels.
    ssd_boxes: SSD boxes.
    ssd_scores: SSD scores.
    ssd_labels: SSD labels.

  Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: Combined outputs.
  """
  if len(deta_boxes) == 0:
    return ssd_boxes, ssd_scores, ssd_labels
  if len(ssd_boxes) == 0:
    return deta_boxes, deta_scores, deta_labels
  boxes = np.concatenate([deta_boxes, ssd_boxes], axis = 0)
  scores = np.concatenate([deta_scores, ssd_scores], axis = 0)
  labels = np.concatenate([deta_labels, ssd_labels], axis = 0)
  return boxes, scores, labels

def build_good_weld_ssd(device):
  """Build the SSDlite weld-region model.

  Args:
    device: Torch device string.

  Returns:
    torch.nn.Module: SSDlite model.
  """
  model = build_ssd_model(
    weights = None,
    weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    num_classes = 2,
  )
  for param in model.backbone.parameters():
    param.requires_grad = False
  model.to(device)
  model.eval()
  return model

def load_good_weld_ssd_from_checkpoint(ssd_checkpoint, device):
  """Load trained SSDlite checkpoint.

  Args:
    ssd_checkpoint: Path to SSD checkpoint.
    device: Torch device string.

  Returns:
    torch.nn.Module: Loaded SSD model.
  """
  model = build_good_weld_ssd(device)
  checkpoint = torch.load(
    ssd_checkpoint,
    map_location = "cpu",
    weights_only = False
  )
  state_dict = checkpoint["model_state_dict"]
  model.load_state_dict(state_dict)
  model.to(device)
  model.eval()
  return model

def predict_good_weld_one(
  ssd_model,
  pil_image,
  threshold,
  device,
  location_iou_threshold,
):
  """Run SSDlite inference for Good Weld fallback.

  Args:
    ssd_model: Loaded SSDlite model.
    pil_image: PIL RGB image.
    threshold: Minimum score threshold.
    device: Torch device string.
    location_iou_threshold: IoU threshold for duplicate grouping.

  Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: Boxes, scores, labels.
  """
  width, height = pil_image.size
  image_rgb = np.array(pil_image)
  image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
  image_tensor = image_tensor.to(device)
  with torch.inference_mode():
    outputs = ssd_model([image_tensor])[0]
  boxes = outputs["boxes"].detach().cpu()
  scores = outputs["scores"].detach().cpu()
  labels = outputs["labels"].detach().cpu()
  keep = (scores >= threshold) & (labels == 1)
  boxes = boxes[keep]
  scores = scores[keep]
  labels = labels[keep]
  if len(scores) == 0:
    return (
      np.zeros((0, 4), dtype = np.float32),
      np.zeros((0,), dtype = np.float32),
      np.zeros((0,), dtype = np.int64),
    )
  boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width - 1)
  boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height - 1)
  boxes, scores, labels = keep_best_per_location(
    boxes = boxes,
    scores = scores,
    labels = labels,
    iou_threshold = location_iou_threshold
  )
  labels = torch.full_like(labels, fill_value = -1)
  return boxes.numpy(), scores.numpy(), labels.numpy()
#
#%% execution loop
#
def main():
  """Watch a folder, run inference, display results, and delete images."""
  args = parse_args()
  checkpoint = resolve_path(args.checkpoint, must_exist = True)
  image_dir = resolve_path(args.image_dir, must_exist = False)
  output_dir = resolve_path(args.output_dir, must_exist = False)
  test_images_dir = resolve_path(args.test_images, must_exist = True)
  if test_images_dir is not None and test_images_dir == image_dir:
    raise ValueError("--test-images must not be the same as --image-dir")
  if not torch.cuda.is_available():
    print("cuda is required but is not available", file = sys.stderr)
    sys.exit(1)
  device = "cuda"
  clear_image_dir(image_dir)
  class_names = load_classes(args.classes)
  model = build_deta_from_checkpoint(checkpoint, device)
  ssd_model = None
  if args.ssd_checkpoint is not None:
    ssd_checkpoint = resolve_path(args.ssd_checkpoint, must_exist = True)
    ssd_model = load_good_weld_ssd_from_checkpoint(
      ssd_checkpoint = ssd_checkpoint,
      device = device
    )
  transform = make_transform(args.image_size)
  window_name = "DETA inference"
  blank = make_blank_image(args.blank_width, args.blank_height)
  test_paths = get_test_image_paths(test_images_dir)
  if test_images_dir is not None and not test_paths:
    print(f"no test images found in: {test_images_dir}")
  next_test_idx = 0
  cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(window_name, args.blank_width, args.blank_height)
  cv2.moveWindow(window_name, args.window_x, args.window_y)
  cv2.imshow(window_name, blank)
  print(f"watching: {image_dir}")
  print("press any key after each result to delete the image and continue")
  print("press Ctrl-C in the terminal to exit")
  try:
    while True:
      wait_result = "key"
      if test_paths:
        path = test_paths[next_test_idx % len(test_paths)]
        next_test_idx += 1
      else:
        paths = get_image_paths(image_dir)
        if not paths:
          show_blank(window_name, blank, args.poll_ms)
          continue
        path = paths[0]
      if not wait_until_file_stable(path, args.stable_wait_ms):
        show_blank(window_name, blank, args.poll_ms)
        continue
      try:
        annotated = process_image(
          path = path,
          model = model,
          ssd_model = ssd_model,
          transform = transform,
          class_names = class_names,
          args = args,
          device = device
        )
        save_annotated_image(
          output_dir = output_dir,
          image_path = path,
          annotated = annotated
        )
        if not test_paths:
          print(f"deleting processed image: {path.name}", flush = True)
          deleted = delete_image(path)
          if not deleted:
            print(
              f"processed image is locked and could not be deleted: {path.name}",
              flush = True
            )
        cv2.imshow(window_name, annotated)
        if test_paths:
          cv2.waitKey(0)
          wait_result = "key"
        else:
          wait_result = wait_for_key_or_new_image(
            window_name = window_name,
            image_dir = image_dir,
            poll_ms = args.poll_ms
          )
      except Exception as exc:
        print(f"failed on {path}: {exc}", flush = True)
        traceback.print_exc()
        if not test_paths:
          print(f"deleting failed image: {path.name}", flush = True)
          deleted = delete_image(path)
          if not deleted:
            print(
              f"leaving failed image in place because it is locked: {path.name}",
              flush = True
            )
          cv2.imshow(window_name, blank)
          cv2.waitKey(args.poll_ms)
          continue
      if test_paths or wait_result == "key":
        cv2.imshow(window_name, blank)
        cv2.waitKey(args.poll_ms)
  except KeyboardInterrupt:
    print("\nexiting")
  finally:
    cv2.destroyAllWindows()
#
#%% run from cli
#
if __name__ == "__main__":
  main()
