#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:50:53 2026

@author: eafpres

Train a Torchvision SSDlite weld-region detector.

This script fine-tunes ssdlite320_mobilenet_v3_large on a
single-class COCO dataset where category id 1 is weld_region.
"""

#
# libraries
#
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
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
    description = "Train SSDlite weld-region detector"
  )
  parser.add_argument(
    "--train-json",
    required = True,
    type = str,
    help = "Path to training COCO JSON"
  )
  parser.add_argument(
    "--val-json",
    required = True,
    type = str,
    help = "Path to validation COCO JSON"
  )
  parser.add_argument(
    "--images-dir",
    required = True,
    type = str,
    help = "Directory containing all image files"
  )
  parser.add_argument(
    "--output-dir",
    required = True,
    type = str,
    help = "Directory for checkpoints"
  )
  parser.add_argument(
    "--epochs",
    type = int,
    default = 20,
    help = "Number of training epochs"
  )
  parser.add_argument(
    "--batch-size",
    type = int,
    default = 4,
    help = "Training batch size"
  )
  parser.add_argument(
    "--val-batch-size",
    type = int,
    default = 2,
    help = "Validation batch size"
  )
  parser.add_argument(
    "--lr",
    type = float,
    default = 1e-4,
    help = "Learning rate"
  )
  parser.add_argument(
    "--weight-decay",
    type = float,
    default = 1e-4,
    help = "Weight decay"
  )
  parser.add_argument(
    "--num-workers",
    type = int,
    default = 4,
    help = "Dataloader workers"
  )
  parser.add_argument(
    "--image-size",
    type = int,
    default = 320,
    help = "Optional resize for input inspection only"
  )
  parser.add_argument(
    "--device",
    type = str,
    default = "cuda",
    choices = ["cuda", "cpu"],
    help = "Training device"
  )
  parser.add_argument(
    "--save-every",
    type = int,
    default = 1,
    help = "Save latest checkpoint every N epochs"
  )
  parser.add_argument(
    "--log-every",
    type = int,
    default = 50,
    help = "Print batch loss every N steps"
  )
  return parser.parse_args()

def load_json(path: Path) -> Dict[str, Any]:
  """Load JSON file.

  Args:
    path: Input path.

  Returns:
    Parsed JSON.
  """
  with path.open("r", encoding = "utf-8") as f:
    return json.load(f)

def collate_fn(
  batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], ...]]:
  """Collate detection batch.

  Args:
    batch: Dataset batch.

  Returns:
    Tuple of images and targets.
  """
  return tuple(zip(*batch))

#
# dataset
#
class CocoWeldRegionDataset(Dataset):
  """Single-class COCO dataset for weld-region detection."""

  def __init__(
    self,
    coco_json: str,
    images_dir: str,
  ) -> None:
    """Initialize dataset.

    Args:
      coco_json: Path to COCO JSON.
      images_dir: Root image directory.
    """
    self.coco_path = Path(coco_json).resolve()
    self.images_dir = Path(images_dir).resolve()
    self.coco = load_json(self.coco_path)
    self.images = self.coco.get("images", [])
    self.categories = self.coco.get("categories", [])
#
# validate one-class schema
#
    if len(self.categories) != 1:
      raise ValueError(
        f"expected exactly 1 category, found {len(self.categories)}"
      )
    cat = self.categories[0]
    if int(cat["id"]) != 1:
      raise ValueError(
        f"expected category id 1 for weld_region, found {cat['id']}"
      )
#
# build annotation index
#
    ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in self.coco.get("annotations", []):
      image_id = int(ann["image_id"])
      ann_by_image.setdefault(image_id, []).append(ann)
    self.ann_by_image = ann_by_image

  def __len__(self) -> int:
    """Return dataset length.

    Returns:
      Number of images.
    """
    return len(self.images)

  def __getitem__(
    self,
    idx: int
  ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return one image and target.

    Args:
      idx: Sample index.

    Returns:
      Image tensor and detection target.
    """
    image_rec = self.images[idx]
    image_id = int(image_rec["id"])
    file_name = str(image_rec["file_name"])
    image_path = self.images_dir / file_name
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
      raise FileNotFoundError(f"could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    anns = self.ann_by_image.get(image_id, [])
    boxes_xyxy: List[List[float]] = []
    labels: List[int] = []
    areas: List[float] = []
    iscrowd: List[int] = []
    for ann in anns:
      x, y, w, h = [float(v) for v in ann["bbox"]]
      if w <= 0 or h <= 0:
        continue
      boxes_xyxy.append([x, y, x + w, y + h])
      labels.append(1)
      areas.append(float(ann.get("area", w * h)))
      iscrowd.append(int(ann.get("iscrowd", 0)))
    if len(boxes_xyxy) == 0:
#
# detection models can handle empty targets, but your current
# converted files should normally have one weld box per image
#
      boxes = torch.zeros((0, 4), dtype = torch.float32)
      labels_t = torch.zeros((0,), dtype = torch.int64)
      areas_t = torch.zeros((0,), dtype = torch.float32)
      iscrowd_t = torch.zeros((0,), dtype = torch.int64)
    else:
      boxes = torch.tensor(boxes_xyxy, dtype = torch.float32)
      labels_t = torch.tensor(labels, dtype = torch.int64)
      areas_t = torch.tensor(areas, dtype = torch.float32)
      iscrowd_t = torch.tensor(iscrowd, dtype = torch.int64)
    target = {
      "boxes": boxes,
      "labels": labels_t,
      "image_id": torch.tensor([image_id], dtype = torch.int64),
      "area": areas_t,
      "iscrowd": iscrowd_t,
    }
    return image, target

#
# model
#
def build_model(num_classes = 2) -> torch.nn.Module:
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
    param.requires_grad = True
  return model

#
# training / validation
#
def move_targets_to_device(
  targets: Tuple[Dict[str, torch.Tensor], ...],
  device: torch.device
) -> List[Dict[str, torch.Tensor]]:
  """Move target dicts to device.

  Args:
    targets: Batch targets.
    device: Torch device.

  Returns:
    Device-mapped targets.
  """
  out: List[Dict[str, torch.Tensor]] = []
  for target in targets:
    out.append(
      {k: v.to(device) for k, v in target.items()}
    )
  return out

def train_one_epoch(
  model: torch.nn.Module,
  loader: DataLoader,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
  epoch_idx: int,
  log_every: int,
) -> float:
  """Train one epoch.

  Args:
    model: Detection model.
    loader: Training loader.
    optimizer: Optimizer.
    device: Torch device.
    epoch_idx: Epoch index.

  Returns:
    Mean training loss.
  """
  model.train()
  loss_sum = 0.0
  n_steps = 0
  for step_idx, (images, targets) in enumerate(loader, start = 1):
    images_d = [img.to(device) for img in images]
    targets_d = move_targets_to_device(targets, device)
    loss_dict = model(images_d, targets_d)
    loss = sum(loss_dict.values())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_value = float(loss.item())
    loss_sum += loss_value
    n_steps += 1
    if (step_idx % log_every) == 0 or step_idx == len(loader):
      print(
        f"epoch {epoch_idx + 1} "
        f"step {step_idx}/{len(loader)} "
        f"train_loss={loss_value:.4f}"
      )
  return loss_sum / max(n_steps, 1)

@torch.no_grad()
def validate_one_epoch(
  model: torch.nn.Module,
  loader: DataLoader,
  device: torch.device,
  epoch_idx: int,
) -> float:
  """Compute validation loss.

  Args:
    model: Detection model.
    loader: Validation loader.
    device: Torch device.
    epoch_idx: Epoch index.

  Returns:
    Mean validation loss.
  """
#
# torchvision detection models only return losses in train mode
# when targets are provided, so use train mode under no_grad
#
  model.train()
  loss_sum = 0.0
  n_steps = 0
  for step_idx, (images, targets) in enumerate(loader, start = 1):
    images_d = [img.to(device) for img in images]
    targets_d = move_targets_to_device(targets, device)
    loss_dict = model(images_d, targets_d)
    loss = sum(loss_dict.values())
    loss_value = float(loss.item())
    loss_sum += loss_value
    n_steps += 1
  return loss_sum / max(n_steps, 1)

def save_checkpoint(
  path: Path,
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  epoch_idx: int,
  train_loss: float,
  val_loss: float,
) -> None:
  """Save checkpoint.

  Args:
    path: Output path.
    model: Detection model.
    optimizer: Optimizer.
    epoch_idx: Epoch index.
    train_loss: Training loss.
    val_loss: Validation loss.
  """
  path.parent.mkdir(parents = True, exist_ok = True)
  torch.save(
    {
      "epoch": epoch_idx,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "train_loss": train_loss,
      "val_loss": val_loss,
      "num_classes": 2,
      "class_names": ["__background__", "weld_region"],
    },
    path,
  )

#
# main
#
def main() -> None:
  """Run training."""
  args = parse_args()
  device = torch.device(args.device)
  if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("cuda requested but not available")
  output_dir = Path(args.output_dir).resolve()
  output_dir.mkdir(parents = True, exist_ok = True)
#
# datasets
#
  train_ds = CocoWeldRegionDataset(
    coco_json = args.train_json,
    images_dir = args.images_dir,
  )
  val_ds = CocoWeldRegionDataset(
    coco_json = args.val_json,
    images_dir = args.images_dir,
  )
  train_loader = DataLoader(
    train_ds,
    batch_size = args.batch_size,
    shuffle = True,
    num_workers = args.num_workers,
    collate_fn = collate_fn,
    pin_memory = (device.type == "cuda"),
  )
  val_loader = DataLoader(
    val_ds,
    batch_size = args.val_batch_size,
    shuffle = False,
    num_workers = args.num_workers,
    collate_fn = collate_fn,
    pin_memory = (device.type == "cuda"),
  )
#
# model / optimizer
#
  model = build_model(num_classes = 2).to(device)
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = args.lr,
    weight_decay = args.weight_decay,
  )
#
# training loop
#
  best_val_loss = math.inf
  for epoch_idx in range(args.epochs):
    train_loss = train_one_epoch(
      model = model,
      loader = train_loader,
      optimizer = optimizer,
      device = device,
      epoch_idx = epoch_idx,
      log_every = args.log_every,
    )
    val_loss = validate_one_epoch(
      model = model,
      loader = val_loader,
      device = device,
      epoch_idx = epoch_idx,
    )
    print(
      f"epoch {epoch_idx + 1} "
      f"train_loss={train_loss:.4f} "
      f"val_loss={val_loss:.4f}"
    )
    if ((epoch_idx + 1) % args.save_every) == 0:
      save_checkpoint(
        path = output_dir / "ssdlite_weld_region_latest.pt",
        model = model,
        optimizer = optimizer,
        epoch_idx = epoch_idx,
        train_loss = train_loss,
        val_loss = val_loss,
      )
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_checkpoint(
        path = output_dir / "ssdlite_weld_region_best.pt",
        model = model,
        optimizer = optimizer,
        epoch_idx = epoch_idx,
        train_loss = train_loss,
        val_loss = val_loss,
      )
      print(
        f"saved new best checkpoint: "
        f"{output_dir / 'ssdlite_weld_region_best.pt'}"
      )

if __name__ == "__main__":
  main()
