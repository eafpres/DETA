#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create targeted COCO augmentations for small damage objects.

The script copies the original COCO dataset into a new self-contained
output directory, adds object-centered zoom crops for selected small or
thin damage annotations, optionally adds reviewed hard-negative crops,
and writes audit CSV files.
"""
#
# libraries
#
import argparse
import copy
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
#
# dataclasses
#
@dataclass
class CocoBox:
  """Single COCO bounding-box annotation.

  Args:
    annotation_id: Source annotation identifier.
    image_id: Source image identifier.
    category_id: COCO category identifier.
    bbox: Bounding box in COCO format [x, y, width, height].
    iscrowd: COCO crowd flag.
  """
  annotation_id: int
  image_id: int
  category_id: int
  bbox: List[float]
  iscrowd: int
@dataclass
class CocoImageRecord:
  """COCO image record with associated annotations.

  Args:
    image_id: Source image identifier.
    file_name: Image path relative to the image root.
    width: Image width in pixels.
    height: Image height in pixels.
    annotations: Bounding-box annotations for the image.
  """
  image_id: int
  file_name: str
  width: int
  height: int
  annotations: List[CocoBox]
@dataclass
class CropResult:
  """Generated crop and adjusted annotations.

  Args:
    image: Cropped and transformed RGB image.
    boxes: Adjusted COCO annotations retained in the crop.
    crop_bbox: Crop window in source-image coordinates.
    zoom_factor: Effective linear zoom relative to the source image.
    flipped: Whether a horizontal flip was applied.
  """
  image: np.ndarray
  boxes: List[CocoBox]
  crop_bbox: List[float]
  zoom_factor: float
  flipped: bool
#
# argument parsing
#
def parse_args() -> argparse.Namespace:
  """Parse command-line arguments.

  Returns:
    Parsed argument namespace.
  """
  parser = argparse.ArgumentParser(
    description = "Create targeted zoom crops for small COCO objects"
  )
  parser.add_argument(
    "--coco-json",
    required = True,
    type = str,
    help = "Path to the source COCO annotations JSON"
  )
  parser.add_argument(
    "--images-dir",
    required = True,
    type = str,
    help = "Root directory containing source images"
  )
  parser.add_argument(
    "--output-dir",
    required = True,
    type = str,
    help = "Output directory for the self-contained augmented dataset"
  )
  parser.add_argument(
    "--output-json",
    type = str,
    default = None,
    help = "Output JSON path; defaults to <output-dir>/annotations.json"
  )
  parser.add_argument(
    "--audit-csv",
    type = str,
    default = None,
    help = "Audit CSV path; defaults to <output-dir>/augmentation_audit.csv"
  )
  parser.add_argument(
    "--summary-csv",
    type = str,
    default = None,
    help = "Summary CSV path; defaults to <output-dir>/augmentation_summary.csv"
  )
  parser.add_argument(
    "--target-classes",
    nargs = "+",
    default = ["Nick", "Scratch", "Hole", "Dent", "Crack"],
    help = "Category names or ids eligible for targeted zoom cropping"
  )
  parser.add_argument(
    "--class-crop-counts",
    nargs = "+",
    default = [
      "Nick=2500",
      "Scratch=1250",
      "Hole=350",
      "Dent=450",
      "Crack=450",
    ],
    help = "Requested crop counts as class=count tokens"
  )
  parser.add_argument(
    "--zoom-ranges",
    nargs = "+",
    default = [
      "Nick=2.0:4.0",
      "Scratch=1.5:3.0",
      "Hole=1.5:3.0",
      "Dent=1.5:3.0",
      "Crack=1.5:3.0",
    ],
    help = "Class-specific linear zoom ranges as class=min:max tokens"
  )
  parser.add_argument(
    "--default-zoom-range",
    nargs = 2,
    type = float,
    default = [1.5, 3.0],
    metavar = ("MIN", "MAX"),
    help = "Fallback zoom range for classes without an explicit range"
  )
  parser.add_argument(
    "--resize-short-side",
    type = int,
    default = 800,
    help = "Reference DETR-style resized short side used for eligibility"
  )
  parser.add_argument(
    "--resize-max-long-side",
    type = int,
    default = 1333,
    help = "Reference DETR-style resized long-side cap"
  )
  parser.add_argument(
    "--small-area-threshold",
    type = float,
    default = 1024.0,
    help = "Eligible resized bbox area threshold; default is 32 squared"
  )
  parser.add_argument(
    "--thin-min-dim-threshold",
    type = float,
    default = 32.0,
    help = "Eligible resized minimum bbox dimension threshold"
  )
  parser.add_argument(
    "--max-crops-per-source-image",
    type = int,
    default = 3,
    help = "Maximum targeted crops generated from one source image"
  )
  parser.add_argument(
    "--max-crops-per-annotation",
    type = int,
    default = 2,
    help = "Maximum targeted crops generated from one source annotation"
  )
  parser.add_argument(
    "--min-neighbor-visibility",
    type = float,
    default = 0.70,
    help = "Minimum visible fraction required to keep neighboring boxes"
  )
  parser.add_argument(
    "--target-margin",
    type = float,
    default = 0.15,
    help = "Minimum target margin as a fraction of target box size"
  )
  parser.add_argument(
    "--center-jitter",
    type = float,
    default = 0.10,
    help = "Crop-center jitter as a fraction of crop width and height"
  )
  parser.add_argument(
    "--min-output-box-area",
    type = float,
    default = 16.0,
    help = "Minimum retained output bbox area"
  )
  parser.add_argument(
    "--horizontal-flip-prob",
    type = float,
    default = 0.50,
    help = "Probability of horizontally flipping each generated crop"
  )
  parser.add_argument(
    "--brightness-limit",
    type = float,
    default = 0.15,
    help = "Maximum random brightness shift as a unit fraction"
  )
  parser.add_argument(
    "--contrast-limit",
    type = float,
    default = 0.15,
    help = "Maximum random contrast shift as a unit fraction"
  )
  parser.add_argument(
    "--saturation-limit",
    type = float,
    default = 0.10,
    help = "Maximum random saturation multiplier shift as a unit fraction"
  )
  parser.add_argument(
    "--jpeg-quality-range",
    nargs = 2,
    type = int,
    default = [80, 100],
    metavar = ("MIN", "MAX"),
    help = "JPEG quality range for generated crop images"
  )
  parser.add_argument(
    "--image-format",
    choices = ["jpg", "png"],
    default = "jpg",
    help = "Generated crop image format"
  )
  parser.add_argument(
    "--hard-negative-csv",
    type = str,
    default = None,
    help = (
      "Optional reviewed false-positive CSV with columns file_name,x,y,"
      "width,height"
    )
  )
  parser.add_argument(
    "--hard-negative-count",
    type = int,
    default = 0,
    help = "Maximum reviewed hard-negative crops to add"
  )
  parser.add_argument(
    "--hard-negative-zoom-range",
    nargs = 2,
    type = float,
    default = [1.5, 3.0],
    metavar = ("MIN", "MAX"),
    help = "Linear zoom range for reviewed hard-negative crops"
  )
  parser.add_argument(
    "--copy-originals",
    dest = "copy_originals",
    action = "store_true",
    help = "Copy original images into the output dataset"
  )
  parser.add_argument(
    "--no-copy-originals",
    dest = "copy_originals",
    action = "store_false",
    help = "Reference original images without copying them"
  )
  parser.set_defaults(copy_originals = True)
  parser.add_argument(
    "--overwrite",
    action = "store_true",
    help = "Replace an existing output directory"
  )
  parser.add_argument(
    "--dry-run",
    action = "store_true",
    help = "Report eligible annotations without writing images"
  )
  parser.add_argument(
    "--seed",
    type = int,
    default = 42,
    help = "Random seed"
  )
  return parser.parse_args()
#
# basic helpers
#
def ensure_dir(path: Path) -> None:
  """Create a directory and its parents when needed.

  Args:
    path: Directory path.
  """
  path.mkdir(parents = True, exist_ok = True)
def load_json(path: Path) -> Dict[str, Any]:
  """Load JSON from disk.

  Args:
    path: JSON file path.

  Returns:
    Parsed JSON dictionary.
  """
  with path.open("r", encoding = "utf-8") as f:
    return json.load(f)
def save_json(data: Dict[str, Any], path: Path) -> None:
  """Write JSON to disk.

  Args:
    data: JSON-compatible dictionary.
    path: Output JSON path.
  """
  ensure_dir(path.parent)
  with path.open("w", encoding = "utf-8") as f:
    json.dump(data, f, indent = 2)
def read_image(path: Path) -> np.ndarray:
  """Read an image as RGB uint8 data.

  Args:
    path: Source image path.

  Returns:
    RGB image array.
  """
  bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if bgr is None:
    raise FileNotFoundError(f"could not read image: {path}")
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
def write_image(
  image: np.ndarray,
  path: Path,
  image_format: str,
  jpeg_quality: int
) -> None:
  """Write an RGB image.

  Args:
    image: RGB uint8 image array.
    path: Output path.
    image_format: Output image format.
    jpeg_quality: JPEG quality used for jpg files.
  """
  ensure_dir(path.parent)
  bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if image_format == "jpg":
    ok = cv2.imwrite(
      str(path),
      bgr,
      [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
  else:
    ok = cv2.imwrite(str(path), bgr)
  if not ok:
    raise IOError(f"failed to write image: {path}")
def parse_key_int_tokens(tokens: Sequence[str]) -> Dict[str, int]:
  """Parse class=count tokens.

  Args:
    tokens: Input tokens.

  Returns:
    Mapping from class token to integer count.
  """
  parsed: Dict[str, int] = {}
  for token in tokens:
    if "=" not in token:
      raise ValueError(f"expected class=count token, received: {token}")
    key, value = token.split("=", 1)
    parsed[key.strip()] = int(value)
  return parsed
def parse_key_range_tokens(tokens: Sequence[str]) -> Dict[str, Tuple[float, float]]:
  """Parse class=min:max tokens.

  Args:
    tokens: Input tokens.

  Returns:
    Mapping from class token to numeric range.
  """
  parsed: Dict[str, Tuple[float, float]] = {}
  for token in tokens:
    if "=" not in token or ":" not in token:
      raise ValueError(f"expected class=min:max token, received: {token}")
    key, value = token.split("=", 1)
    low, high = value.split(":", 1)
    parsed[key.strip()] = (float(low), float(high))
  return parsed
def build_category_lookups(
  categories: Sequence[Dict[str, Any]]
) -> Tuple[Dict[int, str], Dict[str, int]]:
  """Build category lookup dictionaries.

  Args:
    categories: COCO category records.

  Returns:
    Tuple containing id-to-name and lowercase-name-to-id mappings.
  """
  id_to_name: Dict[int, str] = {}
  name_to_id: Dict[str, int] = {}
  for category in categories:
    category_id = int(category["id"])
    name = str(category["name"])
    id_to_name[category_id] = name
    name_to_id[name.lower()] = category_id
  return id_to_name, name_to_id
def resolve_category_token(
  token: str,
  id_to_name: Dict[int, str],
  name_to_id: Dict[str, int]
) -> int:
  """Resolve one category name or id token.

  Args:
    token: Category name or integer id.
    id_to_name: Category-id lookup.
    name_to_id: Lowercase-name lookup.

  Returns:
    Resolved category id.
  """
  stripped = str(token).strip()
  if stripped.lstrip("-").isdigit():
    category_id = int(stripped)
    if category_id not in id_to_name:
      raise ValueError(f"unknown category id: {category_id}")
    return category_id
  category_id = name_to_id.get(stripped.lower())
  if category_id is None:
    raise ValueError(f"unknown category name: {stripped}")
  return category_id
def load_coco_records(
  coco: Dict[str, Any]
) -> Tuple[List[CocoImageRecord], Dict[int, CocoImageRecord]]:
  """Load COCO image records and group annotations by image.

  Args:
    coco: Source COCO dictionary.

  Returns:
    Ordered records and image-id lookup.
  """
  ann_by_image: Dict[int, List[CocoBox]] = defaultdict(list)
  for annotation in coco.get("annotations", []):
    bbox = [float(value) for value in annotation["bbox"]]
    box = CocoBox(
      annotation_id = int(annotation["id"]),
      image_id = int(annotation["image_id"]),
      category_id = int(annotation["category_id"]),
      bbox = bbox,
      iscrowd = int(annotation.get("iscrowd", 0)),
    )
    ann_by_image[box.image_id].append(box)
  records: List[CocoImageRecord] = []
  for image in coco.get("images", []):
    image_id = int(image["id"])
    records.append(
      CocoImageRecord(
        image_id = image_id,
        file_name = str(image["file_name"]),
        width = int(image["width"]),
        height = int(image["height"]),
        annotations = ann_by_image.get(image_id, []),
      )
    )
  return records, {record.image_id: record for record in records}
def model_resize_scale(
  width: float,
  height: float,
  short_side: int,
  max_long_side: int
) -> float:
  """Calculate DETR-style aspect-preserving resize scale.

  Args:
    width: Source image width.
    height: Source image height.
    short_side: Requested resized short side.
    max_long_side: Maximum resized long side.

  Returns:
    Linear resize factor.
  """
  min_side = min(width, height)
  max_side = max(width, height)
  scale = float(short_side) / max(min_side, 1.0)
  if max_side * scale > float(max_long_side):
    scale = float(max_long_side) / max(max_side, 1.0)
  return scale
def intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
  """Calculate intersection area for two COCO boxes.

  Args:
    box_a: First COCO bounding box.
    box_b: Second COCO bounding box.

  Returns:
    Intersection area in pixels squared.
  """
  ax, ay, aw, ah = box_a
  bx, by, bw, bh = box_b
  x1 = max(ax, bx)
  y1 = max(ay, by)
  x2 = min(ax + aw, bx + bw)
  y2 = min(ay + ah, by + bh)
  return max(0.0, x2 - x1) * max(0.0, y2 - y1)
def clip_box_to_crop(
  box: Sequence[float],
  crop: Sequence[float]
) -> Optional[List[float]]:
  """Clip a source box to a crop and translate it to crop coordinates.

  Args:
    box: Source COCO bounding box.
    crop: Crop COCO bounding box.

  Returns:
    Translated clipped box or None when the intersection is empty.
  """
  bx, by, bw, bh = box
  cx, cy, cw, ch = crop
  x1 = max(bx, cx)
  y1 = max(by, cy)
  x2 = min(bx + bw, cx + cw)
  y2 = min(by + bh, cy + ch)
  if x2 <= x1 or y2 <= y1:
    return None
  return [x1 - cx, y1 - cy, x2 - x1, y2 - y1]
def apply_photometric_augmentation(
  image: np.ndarray,
  rng: random.Random,
  brightness_limit: float,
  contrast_limit: float,
  saturation_limit: float
) -> np.ndarray:
  """Apply mild brightness, contrast, and saturation variation.

  Args:
    image: RGB uint8 image.
    rng: Random generator.
    brightness_limit: Maximum brightness shift as a unit fraction.
    contrast_limit: Maximum contrast shift as a unit fraction.
    saturation_limit: Maximum saturation multiplier shift.

  Returns:
    Augmented RGB uint8 image.
  """
  contrast = 1.0 + rng.uniform(-contrast_limit, contrast_limit)
  brightness = 255.0 * rng.uniform(-brightness_limit, brightness_limit)
  adjusted = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255)
  hsv = cv2.cvtColor(adjusted.astype(np.uint8), cv2.COLOR_RGB2HSV)
  saturation = 1.0 + rng.uniform(-saturation_limit, saturation_limit)
  hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation, 0, 255)
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
def horizontal_flip(
  image: np.ndarray,
  boxes: Sequence[CocoBox]
) -> Tuple[np.ndarray, List[CocoBox]]:
  """Flip an image and its COCO boxes horizontally.

  Args:
    image: RGB image.
    boxes: Boxes in crop coordinates.

  Returns:
    Flipped image and adjusted boxes.
  """
  width = image.shape[1]
  out_boxes: List[CocoBox] = []
  for box in boxes:
    x, y, w, h = box.bbox
    out_boxes.append(
      CocoBox(
        annotation_id = box.annotation_id,
        image_id = box.image_id,
        category_id = box.category_id,
        bbox = [float(width) - x - w, y, w, h],
        iscrowd = box.iscrowd,
      )
    )
  return cv2.flip(image, 1), out_boxes
#
# crop generation
#
def choose_crop_window(
  image_width: int,
  image_height: int,
  focus_box: Sequence[float],
  zoom_factor: float,
  target_margin: float,
  center_jitter: float,
  rng: random.Random
) -> Optional[List[float]]:
  """Choose an aspect-preserving crop containing a focus box.

  Args:
    image_width: Source image width.
    image_height: Source image height.
    focus_box: Target box in source coordinates.
    zoom_factor: Requested linear zoom factor.
    target_margin: Required target margin as a fraction of target size.
    center_jitter: Maximum random crop-center jitter fraction.
    rng: Random generator.

  Returns:
    Crop box in source coordinates or None when infeasible.
  """
  tx, ty, tw, th = focus_box
  source_aspect = float(image_width) / max(float(image_height), 1.0)
  crop_width = float(image_width) / max(zoom_factor, 1.0)
  crop_height = float(image_height) / max(zoom_factor, 1.0)
  required_width = tw * (1.0 + 2.0 * target_margin)
  required_height = th * (1.0 + 2.0 * target_margin)
  if required_width > crop_width:
    crop_width = required_width
    crop_height = crop_width / source_aspect
  if required_height > crop_height:
    crop_height = required_height
    crop_width = crop_height * source_aspect
  if crop_width > float(image_width) or crop_height > float(image_height):
    return None
  target_center_x = tx + tw / 2.0
  target_center_y = ty + th / 2.0
  preferred_x = target_center_x - crop_width / 2.0
  preferred_y = target_center_y - crop_height / 2.0
  preferred_x += rng.uniform(-center_jitter, center_jitter) * crop_width
  preferred_y += rng.uniform(-center_jitter, center_jitter) * crop_height
  min_x = max(0.0, tx + tw - crop_width)
  max_x = min(tx, float(image_width) - crop_width)
  min_y = max(0.0, ty + th - crop_height)
  max_y = min(ty, float(image_height) - crop_height)
  if min_x > max_x or min_y > max_y:
    return None
  crop_x = min(max(preferred_x, min_x), max_x)
  crop_y = min(max(preferred_y, min_y), max_y)
  return [crop_x, crop_y, crop_width, crop_height]
def crop_positive_image(
  image: np.ndarray,
  record: CocoImageRecord,
  target_box: CocoBox,
  zoom_factor: float,
  args: argparse.Namespace,
  rng: random.Random
) -> Optional[CropResult]:
  """Generate one target-preserving positive crop.

  Args:
    image: Source RGB image.
    record: Source image record.
    target_box: Annotation that must remain fully visible.
    zoom_factor: Requested linear zoom factor.
    args: Parsed arguments.
    rng: Random generator.

  Returns:
    Crop result or None when a valid crop cannot be generated.
  """
  crop = choose_crop_window(
    image_width = record.width,
    image_height = record.height,
    focus_box = target_box.bbox,
    zoom_factor = zoom_factor,
    target_margin = args.target_margin,
    center_jitter = args.center_jitter,
    rng = rng,
  )
  if crop is None:
    return None
  crop_x, crop_y, crop_width, crop_height = crop
  x1 = int(round(crop_x))
  y1 = int(round(crop_y))
  x2 = int(round(crop_x + crop_width))
  y2 = int(round(crop_y + crop_height))
  x1 = max(0, min(x1, record.width - 1))
  y1 = max(0, min(y1, record.height - 1))
  x2 = max(x1 + 1, min(x2, record.width))
  y2 = max(y1 + 1, min(y2, record.height))
  actual_crop = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
  target_visible = intersection_area(target_box.bbox, actual_crop)
  target_area = target_box.bbox[2] * target_box.bbox[3]
  if target_visible + 1e-6 < target_area:
    return None
  boxes: List[CocoBox] = []
  for box in record.annotations:
    visible_area = intersection_area(box.bbox, actual_crop)
    source_area = box.bbox[2] * box.bbox[3]
    if source_area <= 0:
      continue
    visibility = visible_area / source_area
    if box.annotation_id != target_box.annotation_id:
      if visibility < args.min_neighbor_visibility:
        continue
    translated = clip_box_to_crop(box.bbox, actual_crop)
    if translated is None:
      continue
    if translated[2] * translated[3] < args.min_output_box_area:
      continue
    boxes.append(
      CocoBox(
        annotation_id = box.annotation_id,
        image_id = box.image_id,
        category_id = box.category_id,
        bbox = translated,
        iscrowd = box.iscrowd,
      )
    )
  if not any(box.annotation_id == target_box.annotation_id for box in boxes):
    return None
  cropped = image[y1:y2, x1:x2].copy()
  cropped = apply_photometric_augmentation(
    image = cropped,
    rng = rng,
    brightness_limit = args.brightness_limit,
    contrast_limit = args.contrast_limit,
    saturation_limit = args.saturation_limit,
  )
  flipped = rng.random() < args.horizontal_flip_prob
  if flipped:
    cropped, boxes = horizontal_flip(cropped, boxes)
  effective_zoom = min(
    float(record.width) / max(float(x2 - x1), 1.0),
    float(record.height) / max(float(y2 - y1), 1.0),
  )
  return CropResult(
    image = cropped,
    boxes = boxes,
    crop_bbox = actual_crop,
    zoom_factor = effective_zoom,
    flipped = flipped,
  )
def crop_negative_image(
  image: np.ndarray,
  record: CocoImageRecord,
  focus_box: Sequence[float],
  zoom_factor: float,
  args: argparse.Namespace,
  rng: random.Random
) -> Optional[CropResult]:
  """Generate one reviewed hard-negative crop.

  Args:
    image: Source RGB image.
    record: Source image record.
    focus_box: Reviewed false-positive region.
    zoom_factor: Requested linear zoom factor.
    args: Parsed arguments.
    rng: Random generator.

  Returns:
    Negative crop result or None when any ground-truth box overlaps it.
  """
  crop = choose_crop_window(
    image_width = record.width,
    image_height = record.height,
    focus_box = focus_box,
    zoom_factor = zoom_factor,
    target_margin = args.target_margin,
    center_jitter = args.center_jitter,
    rng = rng,
  )
  if crop is None:
    return None
  crop_x, crop_y, crop_width, crop_height = crop
  x1 = max(0, min(int(round(crop_x)), record.width - 1))
  y1 = max(0, min(int(round(crop_y)), record.height - 1))
  x2 = max(x1 + 1, min(int(round(crop_x + crop_width)), record.width))
  y2 = max(y1 + 1, min(int(round(crop_y + crop_height)), record.height))
  actual_crop = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
  if any(intersection_area(box.bbox, actual_crop) > 0.0 for box in record.annotations):
    return None
  cropped = image[y1:y2, x1:x2].copy()
  cropped = apply_photometric_augmentation(
    image = cropped,
    rng = rng,
    brightness_limit = args.brightness_limit,
    contrast_limit = args.contrast_limit,
    saturation_limit = args.saturation_limit,
  )
  flipped = rng.random() < args.horizontal_flip_prob
  if flipped:
    cropped = cv2.flip(cropped, 1)
  effective_zoom = min(
    float(record.width) / max(float(x2 - x1), 1.0),
    float(record.height) / max(float(y2 - y1), 1.0),
  )
  return CropResult(
    image = cropped,
    boxes = [],
    crop_bbox = actual_crop,
    zoom_factor = effective_zoom,
    flipped = flipped,
  )
#
# output helpers
#
def build_output_dataset(coco: Dict[str, Any]) -> Dict[str, Any]:
  """Initialize output COCO payload.

  Args:
    coco: Source COCO dictionary.

  Returns:
    New COCO dataset dictionary.
  """
  info = copy.deepcopy(coco.get("info", {}))
  source_description = str(info.get("description", "")).strip()
  suffix = "targeted small-object zoom-crop augmentation"
  info["description"] = f"{source_description} | {suffix}".strip(" |")
  return {
    "info": info,
    "licenses": copy.deepcopy(coco.get("licenses", [])),
    "categories": copy.deepcopy(coco.get("categories", [])),
    "images": [],
    "annotations": [],
  }
def append_image(
  dataset: Dict[str, Any],
  image_id: int,
  file_name: str,
  width: int,
  height: int
) -> None:
  """Append one COCO image record.

  Args:
    dataset: Output COCO dataset.
    image_id: Output image id.
    file_name: Relative file path.
    width: Image width.
    height: Image height.
  """
  dataset["images"].append(
    {
      "id": image_id,
      "file_name": file_name,
      "width": width,
      "height": height,
    }
  )
def append_boxes(
  dataset: Dict[str, Any],
  boxes: Sequence[CocoBox],
  image_id: int,
  next_annotation_id: int
) -> int:
  """Append boxes to a COCO dataset.

  Args:
    dataset: Output COCO dataset.
    boxes: Boxes to append.
    image_id: New output image id.
    next_annotation_id: First available output annotation id.

  Returns:
    Next unused annotation id.
  """
  annotation_id = next_annotation_id
  for box in boxes:
    bbox = [round(float(value), 3) for value in box.bbox]
    dataset["annotations"].append(
      {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": int(box.category_id),
        "bbox": bbox,
        "area": round(float(bbox[2] * bbox[3]), 3),
        "iscrowd": int(box.iscrowd),
      }
    )
    annotation_id += 1
  return annotation_id
def copy_originals(
  records: Sequence[CocoImageRecord],
  images_dir: Path,
  output_dir: Path,
  out_dataset: Dict[str, Any],
  next_image_id: int,
  next_annotation_id: int,
  should_copy: bool
) -> Tuple[int, int]:
  """Copy or reference source images and append their annotations.

  Args:
    records: Source image records.
    images_dir: Source image root.
    output_dir: Output image root.
    out_dataset: Output COCO dataset.
    next_image_id: First available output image id.
    next_annotation_id: First available output annotation id.
    should_copy: Whether to copy image files.

  Returns:
    Updated image and annotation ids.
  """
  for index, record in enumerate(records, start = 1):
    source_path = images_dir / record.file_name
    if not source_path.exists():
      raise FileNotFoundError(f"missing source image: {source_path}")
    if should_copy:
      destination = output_dir / record.file_name
      ensure_dir(destination.parent)
      shutil.copy2(source_path, destination)
    append_image(
      dataset = out_dataset,
      image_id = next_image_id,
      file_name = record.file_name,
      width = record.width,
      height = record.height,
    )
    next_annotation_id = append_boxes(
      dataset = out_dataset,
      boxes = record.annotations,
      image_id = next_image_id,
      next_annotation_id = next_annotation_id,
    )
    next_image_id += 1
    if index % 500 == 0:
      print(f"copied or referenced {index} original images")
  return next_image_id, next_annotation_id
def audit_row(
  kind: str,
  output_file_name: str,
  record: CocoImageRecord,
  target_box: Optional[CocoBox],
  target_name: str,
  crop: CropResult,
  resized_target_width: Optional[float],
  resized_target_height: Optional[float]
) -> Dict[str, Any]:
  """Build one augmentation audit row.

  Args:
    kind: Augmentation type.
    output_file_name: Generated relative output file path.
    record: Source image record.
    target_box: Source target annotation or None.
    target_name: Target class or hard-negative label.
    crop: Generated crop result.
    resized_target_width: Reference resized target width.
    resized_target_height: Reference resized target height.

  Returns:
    Audit row dictionary.
  """
  crop_x, crop_y, crop_width, crop_height = crop.crop_bbox
  target_bbox = target_box.bbox if target_box is not None else [None] * 4
  return {
    "kind": kind,
    "output_file_name": output_file_name,
    "source_image_id": record.image_id,
    "source_file_name": record.file_name,
    "source_annotation_id": (
      target_box.annotation_id if target_box is not None else ""
    ),
    "target_class": target_name,
    "source_target_x": target_bbox[0],
    "source_target_y": target_bbox[1],
    "source_target_width": target_bbox[2],
    "source_target_height": target_bbox[3],
    "reference_resized_target_width": resized_target_width,
    "reference_resized_target_height": resized_target_height,
    "crop_x": crop_x,
    "crop_y": crop_y,
    "crop_width": crop_width,
    "crop_height": crop_height,
    "effective_zoom": crop.zoom_factor,
    "horizontal_flip": int(crop.flipped),
    "retained_annotations": len(crop.boxes),
  }
def save_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
  """Write dictionaries to a CSV file.

  Args:
    rows: Row dictionaries.
    path: Output CSV path.
  """
  ensure_dir(path.parent)
  if not rows:
    with path.open("w", encoding = "utf-8", newline = "") as f:
      f.write("")
    return
  fieldnames = list(rows[0].keys())
  with path.open("w", encoding = "utf-8", newline = "") as f:
    writer = csv.DictWriter(f, fieldnames = fieldnames)
    writer.writeheader()
    writer.writerows(rows)
def load_hard_negative_rows(path: Path) -> List[Dict[str, Any]]:
  """Load reviewed hard-negative regions from CSV.

  Args:
    path: CSV path.

  Returns:
    Hard-negative row dictionaries.
  """
  required = {"file_name", "x", "y", "width", "height"}
  with path.open("r", encoding = "utf-8-sig", newline = "") as f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
      raise ValueError("hard-negative CSV has no header")
    missing = required.difference(reader.fieldnames)
    if missing:
      raise ValueError(
        "hard-negative CSV missing columns: " + ", ".join(sorted(missing))
      )
    return [dict(row) for row in reader]
#
# main workflow
#
def main() -> None:
  """Run targeted small-object augmentation."""
  args = parse_args()
  rng = random.Random(args.seed)
  np.random.seed(args.seed)
  coco_json = Path(args.coco_json).resolve()
  images_dir = Path(args.images_dir).resolve()
  output_dir = Path(args.output_dir).resolve()
  output_json = (
    Path(args.output_json).resolve()
    if args.output_json is not None
    else output_dir / "annotations.json"
  )
  audit_csv = (
    Path(args.audit_csv).resolve()
    if args.audit_csv is not None
    else output_dir / "augmentation_audit.csv"
  )
  summary_csv = (
    Path(args.summary_csv).resolve()
    if args.summary_csv is not None
    else output_dir / "augmentation_summary.csv"
  )
  if not coco_json.exists():
    raise FileNotFoundError(f"COCO JSON not found: {coco_json}")
  if not images_dir.exists():
    raise FileNotFoundError(f"images directory not found: {images_dir}")
  if output_dir == images_dir:
    raise ValueError("--output-dir must differ from --images-dir")
  if output_dir.exists() and any(output_dir.iterdir()):
    if args.overwrite and not args.dry_run:
      shutil.rmtree(output_dir)
    elif not args.dry_run:
      raise FileExistsError(
        f"output directory is not empty: {output_dir}; use --overwrite"
      )
  coco = load_json(coco_json)
  categories = coco.get("categories", [])
  id_to_name, name_to_id = build_category_lookups(categories)
  target_ids = {
    resolve_category_token(token, id_to_name, name_to_id)
    for token in args.target_classes
  }
  crop_count_tokens = parse_key_int_tokens(args.class_crop_counts)
  zoom_range_tokens = parse_key_range_tokens(args.zoom_ranges)
  requested_counts: Dict[int, int] = {}
  zoom_ranges: Dict[int, Tuple[float, float]] = {}
  for token, count in crop_count_tokens.items():
    category_id = resolve_category_token(token, id_to_name, name_to_id)
    requested_counts[category_id] = count
  for token, zoom_range in zoom_range_tokens.items():
    category_id = resolve_category_token(token, id_to_name, name_to_id)
    zoom_ranges[category_id] = zoom_range
  records, records_by_id = load_coco_records(coco)
  records_by_file = {record.file_name: record for record in records}
  eligible_by_category: Dict[int, List[Tuple[CocoImageRecord, CocoBox, float, float]]] = defaultdict(list)
  for record in records:
    scale = model_resize_scale(
      width = record.width,
      height = record.height,
      short_side = args.resize_short_side,
      max_long_side = args.resize_max_long_side,
    )
    for box in record.annotations:
      if box.category_id not in target_ids:
        continue
      resized_width = box.bbox[2] * scale
      resized_height = box.bbox[3] * scale
      resized_area = resized_width * resized_height
      is_small = resized_area < args.small_area_threshold
      is_thin = min(resized_width, resized_height) < args.thin_min_dim_threshold
      if is_small or is_thin:
        eligible_by_category[box.category_id].append(
          (record, box, resized_width, resized_height)
        )
  print(f"source images: {len(records)}")
  print(f"source annotations: {len(coco.get('annotations', []))}")
  print("eligible targeted annotations:")
  for category_id in sorted(target_ids, key = lambda value: id_to_name[value]):
    print(f"  {id_to_name[category_id]}: {len(eligible_by_category[category_id])}")
  if args.dry_run:
    return
  ensure_dir(output_dir)
  out_dataset = build_output_dataset(coco)
  next_image_id = 1
  next_annotation_id = 1
  next_image_id, next_annotation_id = copy_originals(
    records = records,
    images_dir = images_dir,
    output_dir = output_dir,
    out_dataset = out_dataset,
    next_image_id = next_image_id,
    next_annotation_id = next_annotation_id,
    should_copy = args.copy_originals,
  )
  audit_rows: List[Dict[str, Any]] = []
  generated_counts: Counter = Counter()
  source_image_counts: Counter = Counter()
  annotation_counts: Counter = Counter()
  image_cache: Dict[int, np.ndarray] = {}
  generated_index = 1
  for category_id, requested_count in requested_counts.items():
    candidates = list(eligible_by_category.get(category_id, []))
    rng.shuffle(candidates)
    if not candidates:
      print(f"warning: no eligible candidates for {id_to_name[category_id]}")
      continue
    max_attempts = max(1000, requested_count * 100)
    attempts = 0
    candidate_index = 0
    while generated_counts[category_id] < requested_count and attempts < max_attempts:
      attempts += 1
      record, target_box, resized_width, resized_height = candidates[
        candidate_index % len(candidates)
      ]
      candidate_index += 1
      if source_image_counts[record.image_id] >= args.max_crops_per_source_image:
        continue
      if annotation_counts[target_box.annotation_id] >= args.max_crops_per_annotation:
        continue
      image = image_cache.get(record.image_id)
      if image is None:
        image = read_image(images_dir / record.file_name)
        image_cache[record.image_id] = image
      zoom_low, zoom_high = zoom_ranges.get(
        category_id,
        tuple(args.default_zoom_range),
      )
      zoom_factor = rng.uniform(zoom_low, zoom_high)
      crop = crop_positive_image(
        image = image,
        record = record,
        target_box = target_box,
        zoom_factor = zoom_factor,
        args = args,
        rng = rng,
      )
      if crop is None:
        continue
      out_rel_path = (
        f"targeted_zoom/{id_to_name[category_id].replace(' ', '_')}_"
        f"{generated_index:06d}.{args.image_format}"
      )
      generated_index += 1
      jpeg_quality = rng.randint(
        min(args.jpeg_quality_range),
        max(args.jpeg_quality_range),
      )
      write_image(
        image = crop.image,
        path = output_dir / out_rel_path,
        image_format = args.image_format,
        jpeg_quality = jpeg_quality,
      )
      append_image(
        dataset = out_dataset,
        image_id = next_image_id,
        file_name = out_rel_path,
        width = crop.image.shape[1],
        height = crop.image.shape[0],
      )
      next_annotation_id = append_boxes(
        dataset = out_dataset,
        boxes = crop.boxes,
        image_id = next_image_id,
        next_annotation_id = next_annotation_id,
      )
      audit_rows.append(
        audit_row(
          kind = "targeted_positive",
          output_file_name = out_rel_path,
          record = record,
          target_box = target_box,
          target_name = id_to_name[category_id],
          crop = crop,
          resized_target_width = resized_width,
          resized_target_height = resized_height,
        )
      )
      next_image_id += 1
      generated_counts[category_id] += 1
      source_image_counts[record.image_id] += 1
      annotation_counts[target_box.annotation_id] += 1
      if sum(generated_counts.values()) % 500 == 0:
        print(f"generated {sum(generated_counts.values())} positive crops")
    if generated_counts[category_id] < requested_count:
      print(
        f"warning: generated {generated_counts[category_id]} of "
        f"{requested_count} requested crops for {id_to_name[category_id]}"
      )
  hard_negative_generated = 0
  if args.hard_negative_csv is not None and args.hard_negative_count > 0:
    hard_negative_rows = load_hard_negative_rows(
      Path(args.hard_negative_csv).resolve()
    )
    rng.shuffle(hard_negative_rows)
    for row in hard_negative_rows:
      if hard_negative_generated >= args.hard_negative_count:
        break
      file_name = str(row["file_name"])
      record = records_by_file.get(file_name)
      if record is None:
        print(f"warning: hard-negative image not found in COCO JSON: {file_name}")
        continue
      image = image_cache.get(record.image_id)
      if image is None:
        image = read_image(images_dir / record.file_name)
        image_cache[record.image_id] = image
      focus_box = [
        float(row["x"]),
        float(row["y"]),
        float(row["width"]),
        float(row["height"]),
      ]
      zoom_factor = rng.uniform(
        min(args.hard_negative_zoom_range),
        max(args.hard_negative_zoom_range),
      )
      crop = crop_negative_image(
        image = image,
        record = record,
        focus_box = focus_box,
        zoom_factor = zoom_factor,
        args = args,
        rng = rng,
      )
      if crop is None:
        print(f"warning: rejected overlapping hard-negative row: {file_name}")
        continue
      out_rel_path = f"hard_negative/hard_negative_{generated_index:06d}.{args.image_format}"
      generated_index += 1
      jpeg_quality = rng.randint(
        min(args.jpeg_quality_range),
        max(args.jpeg_quality_range),
      )
      write_image(
        image = crop.image,
        path = output_dir / out_rel_path,
        image_format = args.image_format,
        jpeg_quality = jpeg_quality,
      )
      append_image(
        dataset = out_dataset,
        image_id = next_image_id,
        file_name = out_rel_path,
        width = crop.image.shape[1],
        height = crop.image.shape[0],
      )
      audit_rows.append(
        audit_row(
          kind = "reviewed_hard_negative",
          output_file_name = out_rel_path,
          record = record,
          target_box = None,
          target_name = "hard_negative",
          crop = crop,
          resized_target_width = None,
          resized_target_height = None,
        )
      )
      next_image_id += 1
      hard_negative_generated += 1
  summary_rows: List[Dict[str, Any]] = []
  for category_id in sorted(requested_counts, key = lambda value: id_to_name[value]):
    summary_rows.append(
      {
        "target_class": id_to_name[category_id],
        "eligible_annotations": len(eligible_by_category.get(category_id, [])),
        "requested_crops": requested_counts[category_id],
        "generated_crops": generated_counts[category_id],
      }
    )
  summary_rows.append(
    {
      "target_class": "reviewed_hard_negative",
      "eligible_annotations": "",
      "requested_crops": args.hard_negative_count,
      "generated_crops": hard_negative_generated,
    }
  )
  save_json(out_dataset, output_json)
  save_csv(audit_rows, audit_csv)
  save_csv(summary_rows, summary_csv)
  print(f"wrote output images under: {output_dir}")
  print(f"wrote COCO JSON: {output_json}")
  print(f"wrote audit CSV: {audit_csv}")
  print(f"wrote summary CSV: {summary_csv}")
  print(f"output images: {len(out_dataset['images'])}")
  print(f"output annotations: {len(out_dataset['annotations'])}")
  print(f"generated positive crops: {sum(generated_counts.values())}")
  print(f"generated hard-negative crops: {hard_negative_generated}")
if __name__ == "__main__":
  main()
