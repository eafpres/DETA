#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:06:00 2026

@author: eafpres

Visualize SSDlite weld-region predictions against COCO ground truth."""
#
# libraries
#
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
#
# helpers
#
def parse_args() -> argparse.Namespace:
  """Parse command-line arguments.

  Returns:
    Parsed args.
  """
  parser = argparse.ArgumentParser(
    description = "Visualize SSDlite weld-region predictions"
  )
  parser.add_argument(
    "--coco-json",
    required = True,
    type = str,
    help = "Path to COCO json used for visualization"
  )
  parser.add_argument(
    "--images-dir",
    required = True,
    type = str,
    help = "Root directory containing image files"
  )
  parser.add_argument(
    "--checkpoint",
    required = True,
    type = str,
    help = "Path to trained SSDlite checkpoint"
  )
  parser.add_argument(
    "--device",
    type = str,
    default = "cuda",
    choices = ["cuda", "cpu"],
    help = "Inference device"
  )
  parser.add_argument(
    "--score-thresh",
    type = float,
    default = 0.3,
    help = "Minimum score for predicted boxes"
  )
  parser.add_argument(
    "--max-detections",
    type = int,
    default = 5,
    help = "Maximum predicted boxes to draw"
  )
  parser.add_argument(
    "--num-images",
    type = int,
    default = 12,
    help = "Number of images to visualize"
  )
  parser.add_argument(
    "--random-sample",
    action = "store_true",
    help = "Sample images randomly instead of taking the first N"
  )
  parser.add_argument(
    "--seed",
    type = int,
    default = 42,
    help = "Random seed for sampling"
  )
  parser.add_argument(
    "--only-misses",
    action = "store_true",
    help = (
      "Only show images where there is ground truth but no prediction "
      "above threshold"
    )
  )
  parser.add_argument(
    "--only-false-positives",
    action = "store_true",
    help = (
      "Only show images where there is prediction above threshold but "
      "no ground truth"
    )
  )
  parser.add_argument(
    "--image-id",
    type = int,
    nargs = "+",
    default = None,
    help = "Optional specific image ids to visualize"
  )
  parser.add_argument(
    "--save-dir",
    type = str,
    default = None,
    help = "Optional directory to save rendered figures"
  )
  return parser.parse_args()

def load_json(path: Path) -> Dict[str, Any]:
  """Load JSON file.

  Args:
    path: JSON path.

  Returns:
    Parsed JSON.
  """
  with path.open("r", encoding = "utf-8") as f:
    return json.load(f)

def build_model(num_classes: int = 2) -> torch.nn.Module:
  """Build SSDlite model.

  Args:
    num_classes: Number of classes including background.

  Returns:
    Torchvision detection model.
  """
  model = ssdlite320_mobilenet_v3_large(
    weights = None,
    weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    num_classes = num_classes,
  )
  for param in model.backbone.parameters():
    param.requires_grad = False
  return model

def load_checkpoint_model(
  checkpoint_path: Path,
  device: torch.device,
) -> torch.nn.Module:
  """Load trained SSDlite checkpoint.

  Args:
    checkpoint_path: Path to checkpoint.
    device: Inference device.

  Returns:
    Loaded model in eval mode.
  """
  model = build_model(num_classes = 2)
  ckpt = torch.load(checkpoint_path, map_location = device)
  model.load_state_dict(ckpt["model_state_dict"])
  model.to(device)
  model.eval()
  return model

def build_annotation_index(
  coco: Dict[str, Any]
) -> Dict[int, List[Dict[str, Any]]]:
  """Build image_id to annotations map.

  Args:
    coco: COCO dataset.

  Returns:
    Annotation index.
  """
  ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
  for ann in coco.get("annotations", []):
    image_id = int(ann["image_id"])
    ann_by_image.setdefault(image_id, []).append(ann)
  return ann_by_image

def read_image_rgb(image_path: Path) -> np.ndarray:
  """Read image as RGB.

  Args:
    image_path: Image path.

  Returns:
    RGB uint8 image.
  """
  image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
  if image_bgr is None:
    raise FileNotFoundError(f"could not read image: {image_path}")
  return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def image_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
  """Convert RGB image to float tensor.

  Args:
    image_rgb: RGB uint8 image.

  Returns:
    Float tensor in CHW format scaled to [0, 1].
  """
  return torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

@torch.no_grad()
def predict_image(
  model: torch.nn.Module,
  image_rgb: np.ndarray,
  device: torch.device,
  score_thresh: float,
  max_detections: int,
) -> Dict[str, np.ndarray]:
  """Run SSDlite inference on one image.

  Args:
    model: Trained model.
    image_rgb: RGB uint8 image.
    device: Inference device.
    score_thresh: Score threshold.
    max_detections: Max number of detections to keep.

  Returns:
    Dict with boxes, scores, labels after filtering.
  """
  image_tensor = image_to_tensor(image_rgb).to(device)
  pred = model([image_tensor])[0]
  boxes = pred["boxes"].detach().cpu().numpy()
  scores = pred["scores"].detach().cpu().numpy()
  labels = pred["labels"].detach().cpu().numpy()
  keep = [
    i for i, (score, label) in enumerate(zip(scores, labels))
    if float(score) >= score_thresh and int(label) == 1
  ]
  keep = keep[:max_detections]
  return {
    "boxes": boxes[keep] if len(keep) > 0 else np.zeros((0, 4)),
    "scores": scores[keep] if len(keep) > 0 else np.zeros((0,)),
    "labels": labels[keep] if len(keep) > 0 else np.zeros((0,)),
  }

def coco_bbox_to_xyxy(bbox: List[float]) -> List[float]:
  """Convert COCO bbox to xyxy.

  Args:
    bbox: COCO bbox [x, y, w, h].

  Returns:
    xyxy box.
  """
  x, y, w, h = [float(v) for v in bbox]
  return [x, y, x + w, y + h]

def draw_gt_boxes(
  ax: Any,
  anns: List[Dict[str, Any]],
) -> None:
  """Draw ground-truth boxes.

  Args:
    ax: Matplotlib axis.
    anns: COCO annotations.
  """
  for ann in anns:
    x, y, w, h = [float(v) for v in ann["bbox"]]
    rect = patches.Rectangle(
      (x, y),
      w,
      h,
      linewidth = 2.0,
      edgecolor = "lime",
      facecolor = "none",
    )
    ax.add_patch(rect)
    ax.text(
      x,
      max(0.0, y - 5.0),
      "GT",
      color = "lime",
      fontsize = 9,
      bbox = {
        "facecolor": "black",
        "alpha": 0.5,
        "pad": 1.5,
      },
    )

def draw_pred_boxes(
  ax: Any,
  pred: Dict[str, np.ndarray],
) -> None:
  """Draw predicted boxes.

  Args:
    ax: Matplotlib axis.
    pred: Prediction dict.
  """
  boxes = pred["boxes"]
  scores = pred["scores"]
  for box, score in zip(boxes, scores):
    x0, y0, x1, y1 = [float(v) for v in box.tolist()]
    rect = patches.Rectangle(
      (x0, y0),
      x1 - x0,
      y1 - y0,
      linewidth = 2.0,
      edgecolor = "red",
      facecolor = "none",
      linestyle = "--",
    )
    ax.add_patch(rect)
    ax.text(
      x0,
      min(y1 + 12.0, ax.get_ylim()[0] if False else y0 + 12.0),
      f"Pred {score:.2f}",
      color = "red",
      fontsize = 9,
      bbox = {
        "facecolor": "white",
        "alpha": 0.7,
        "pad": 1.5,
      },
    )

def should_keep_image(
  anns: List[Dict[str, Any]],
  pred: Dict[str, np.ndarray],
  only_misses: bool,
  only_false_positives: bool,
) -> bool:
  """Decide whether an image should be displayed.

  Args:
    anns: Ground-truth annotations.
    pred: Prediction dict.
    only_misses: Whether to keep only misses.
    only_false_positives: Whether to keep only false positives.

  Returns:
    True if the image should be shown.
  """
  has_gt = len(anns) > 0
  has_pred = len(pred["boxes"]) > 0
  if only_misses:
    return has_gt and not has_pred
  if only_false_positives:
    return (not has_gt) and has_pred
  return True

def plot_one_image(
  fig: Any,
  ax: Any,
  image_rgb: np.ndarray,
  anns: List[Dict[str, Any]],
  pred: Dict[str, np.ndarray],
  title: str,
  save_path: Optional[Path] = None,
) -> None:
  """Plot one image with GT and predictions.

  Args:
    fig: Matplotlib figure.
    ax: Matplotlib axis.
    image_rgb: RGB image.
    anns: Ground-truth annotations.
    pred: Prediction dict.
    title: Plot title.
    save_path: Optional save path.
  """
  ax.clear()
  ax.imshow(image_rgb)
  draw_gt_boxes(ax, anns)
  draw_pred_boxes(ax, pred)
  ax.set_title(title)
  ax.axis("off")
  fig.tight_layout()
  if save_path is not None:
    save_path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(save_path, bbox_inches = "tight", dpi = 150)
  else:
    fig.canvas.draw_idle()
    plt.pause(0.001)

def select_image_records(
  images: List[Dict[str, Any]],
  image_ids: Optional[List[int]],
  num_images: int,
  random_sample: bool,
  seed: int,
) -> List[Dict[str, Any]]:
  """Select image records to visualize.

  Args:
    images: COCO image records.
    image_ids: Optional explicit ids.
    num_images: Number of images to select.
    random_sample: Whether to sample randomly.
    seed: Random seed.

  Returns:
    Selected image records.
  """
  if image_ids is not None:
    wanted = set(int(v) for v in image_ids)
    return [img for img in images if int(img["id"]) in wanted]
  images_sorted = sorted(images, key = lambda x: int(x["id"]))
  if random_sample:
    rng = random.Random(seed)
    return rng.sample(images_sorted, k = min(num_images, len(images_sorted)))
  return images_sorted[:num_images]

def main() -> None:
  """Run visualization."""
  args = parse_args()
  device = torch.device(args.device)
  if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("cuda requested but not available")
  coco_json = Path(args.coco_json).resolve()
  images_dir = Path(args.images_dir).resolve()
  checkpoint = Path(args.checkpoint).resolve()
  save_dir = (
    Path(args.save_dir).resolve()
    if args.save_dir is not None
    else None
  )
  coco = load_json(coco_json)
  ann_by_image = build_annotation_index(coco)
  model = load_checkpoint_model(checkpoint, device)
  image_records = select_image_records(
    images = coco.get("images", []),
    image_ids = args.image_id,
    num_images = args.num_images,
    random_sample = args.random_sample,
    seed = args.seed,
  )
  fig = None
  ax = None
  if save_dir is None:
    plt.ion()
    fig, ax = plt.subplots(figsize = (10, 10))
  shown = 0
  for image_rec in image_records:
    image_id = int(image_rec["id"])
    file_name = str(image_rec["file_name"])
    image_path = images_dir / file_name
    try:
      image_rgb = read_image_rgb(image_path)
    except FileNotFoundError as exc:
      print(str(exc))
      continue
    anns = ann_by_image.get(image_id, [])
    pred = predict_image(
      model = model,
      image_rgb = image_rgb,
      device = device,
      score_thresh = args.score_thresh,
      max_detections = args.max_detections,
    )
    if not should_keep_image(
      anns = anns,
      pred = pred,
      only_misses = args.only_misses,
      only_false_positives = args.only_false_positives,
    ):
      continue
    title = (
      f"id={image_id} "
      f"gt={len(anns)} "
      f"pred={len(pred['boxes'])} "
      f"{file_name}"
    )
    save_path = None
    if save_dir is not None:
      save_path = save_dir / f"viz_{image_id:06d}.png"
    if save_dir is not None:
      temp_fig, temp_ax = plt.subplots(figsize = (10, 10))
      plot_one_image(
        fig = temp_fig,
        ax = temp_ax,
        image_rgb = image_rgb,
        anns = anns,
        pred = pred,
        title = title,
        save_path = save_path,
      )
      plt.close(temp_fig)
    else:
      plot_one_image(
        fig = fig,
        ax = ax,
        image_rgb = image_rgb,
        anns = anns,
        pred = pred,
        title = title,
        save_path = None,
      )
      input("Press Enter for next image...")
    shown += 1
    if shown >= args.num_images:
      break
  print(f"visualized images: {shown}")

if __name__ == "__main__":
  main()
