#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:13:31 2026

@author: eafpres

COCO image augmentation with optional mosaic synthesis.

This script reads a COCO detection dataset, applies bbox-aware
Albumentations transforms to individual images, optionally builds
2..N-image mosaics, and writes a new COCO dataset with updated
annotations.

The workflow is designed for datasets where images may contain one or
more objects, but it is especially convenient for small datasets with
one object per image.
"""
#
# libraries
#
import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import albumentations as A
import cv2
import numpy as np
import shutil
#
# constants
#
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
#
# dataclasses
#
@dataclass
class CocoBox:
  """Single COCO bbox annotation.

  Args:
    bbox: Bounding box in COCO format [x, y, width, height].
    category_id: COCO category id.
    annotation_id: Source annotation id.
    iscrowd: COCO iscrowd value.
    area: COCO area value.
    segmentation: Optional segmentation payload copied through when
      present. This script does not geometrically transform
      segmentation polygons.
    source_image_id: Source image id for provenance.
  """
  bbox: List[float]
  category_id: int
  annotation_id: int
  iscrowd: int
  area: float
  segmentation: Any
  source_image_id: int
@dataclass
class CocoImageRecord:
  """Image record plus loaded annotations.

  Args:
    image_id: COCO image id.
    file_name: Relative file name from images root.
    width: Image width.
    height: Image height.
    annotations: List of image annotations.
  """
  image_id: int
  file_name: str
  width: int
  height: int
  annotations: List[CocoBox]
#
# helpers
#
def parse_args() -> argparse.Namespace:
  """Parse command-line arguments.

  Returns:
    Parsed namespace.
  """
  parser = argparse.ArgumentParser(
    description = "Augment a COCO dataset with Albumentations and mosaic"
  )
  parser.add_argument(
    "--coco-json",
    required = True,
    type = str,
    help = "Path to source COCO annotations JSON"
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
    help = "Output directory for augmented dataset"
  )
  parser.add_argument(
    "--copies-per-image",
    type = int,
    default = 3,
    help = "Number of augmented single-image copies per source image"
  )
  parser.add_argument(
    "--mosaic-copies",
    type = float,
    default = 0.15,
    help = "Average number of mosaic outputs to generate per source image"
  )
  parser.add_argument(
    "--enable-mosaic",
    action = "store_true",
    help = "Enable mosaic synthesis"
  )
  parser.add_argument(
    "--min-mosaic-images",
    type = int,
    default = 2,
    help = "Minimum number of source images per mosaic"
  )
  parser.add_argument(
    "--max-mosaic-images",
    type = int,
    default = 2,
    help = "Maximum number of source images per mosaic"
  )
  parser.add_argument(
    "--mosaic-size",
    type = int,
    nargs = 2,
    default = [1600, 800],
    metavar = ("WIDTH", "HEIGHT"),
    help = "Output mosaic size as width height"
  )
  parser.add_argument(
    "--seed",
    type = int,
    default = 42,
    help = "Random seed"
  )
  parser.add_argument(
    "--min-area",
    type = float,
    default = 16.0,
    help = "Minimum bbox area to keep after transforms"
  )
  parser.add_argument(
    "--min-visibility",
    type = float,
    default = 0.30,
    help = "Minimum visible bbox fraction to keep after transforms"
  )
  parser.add_argument(
    "--max-trials-per-output",
    type = int,
    default = 20,
    help = "Retry count when a transform drops all boxes"
  )
  parser.add_argument(
    "--jpeg-quality",
    type = int,
    default = 95,
    help = "JPEG quality for written jpg files"
  )
  parser.add_argument(
    "--image-format",
    type = str,
    default = "jpg",
    choices = ["jpg", "png"],
    help = "Output image format"
  )
  parser.add_argument(
    "--allow-negative-samples",
    action = "store_true",
    help = "Allow transformed outputs that lose all boxes"
  )
  parser.add_argument(
    "--organize-by-source-folder",
    action = "store_true",
    help = (
      "Write output images into subfolders based on the original image "
      "folder"
    )
  )
  parser.add_argument(
    "--include-originals",
    action = "store_true",
    help = "Copy original images and annotations into the output dataset"
  )
  parser.add_argument(
    "--classes-to-augment",
    type = str,
    nargs = "+",
    default = None,
    help = (
      "Only augment images whose annotations include one of these "
      "category names or category ids"
    )
  )
  parser.add_argument(
    "--append-output",
    action = "store_true",
    help = "Append to an existing output dataset instead of replacing it"
  )
  parser.add_argument(
    "--mosaics-only",
    action = "store_true",
    help = (
      "Skip single-image augmentation and build mosaics only from "
      "existing images referenced by --output-json and found under "
      "--images-dir"
    )
  )
  parser.add_argument(
    "--output-json",
    type = str,
    default = None,
    help = (
      "Path to output COCO annotations JSON. Defaults to "
      "<output-dir>/annotations.json"
    )
  )
  return parser.parse_args()
def ensure_dir(path: Path) -> None:
  """Create directory if missing.

  Args:
    path: Directory path.
  """
  path.mkdir(parents = True, exist_ok = True)

def load_json(path: Path) -> Dict[str, Any]:
  """Load JSON from disk.

  Args:
    path: JSON path.

  Returns:
    Parsed dict.
  """
  with path.open("r", encoding = "utf-8") as f:
    return json.load(f)

def save_json(data: Dict[str, Any], path: Path) -> None:
  """Write JSON to disk.

  Args:
    data: Payload to write.
    path: Output path.
  """
  with path.open("w", encoding = "utf-8") as f:
    json.dump(data, f, indent = 2)

def load_existing_output_dataset(
  output_json_path: Path
) -> Optional[Dict[str, Any]]:
  """Load an existing output COCO dataset when present.

  Args:
    output_json_path: Path to output annotations JSON.

  Returns:
    Existing dataset dict or None.
  """
  if not output_json_path.exists():
    return None
  with output_json_path.open("r", encoding = "utf-8") as f:
    return json.load(f)

def get_next_ids_from_dataset(
  dataset: Dict[str, Any]
) -> Tuple[int, int]:
  """Return next image and annotation ids from an existing dataset.

  Args:
    dataset: Existing COCO dataset.

  Returns:
    Tuple of next image id and next annotation id.
  """
  image_ids = [int(img["id"]) for img in dataset.get("images", [])]
  ann_ids = [int(ann["id"]) for ann in dataset.get("annotations", [])]
  next_image_id = (max(image_ids) + 1) if image_ids else 1
  next_annotation_id = (max(ann_ids) + 1) if ann_ids else 1
  return next_image_id, next_annotation_id

def get_next_running_index_from_dataset(
  dataset: Dict[str, Any]
) -> int:
  """Infer next filename index from existing image file names.

  Args:
    dataset: Existing COCO dataset.

  Returns:
    Next running index.
  """
  max_idx = 0
  for image_rec in dataset.get("images", []):
    file_name = Path(str(image_rec.get("file_name", ""))).name
    stem = Path(file_name).stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
      continue
    suffix = parts[1]
    if suffix.isdigit():
      max_idx = max(max_idx, int(suffix))
  return max_idx + 1

def build_existing_file_name_set(
  dataset: Dict[str, Any]
) -> set:
  """Return the set of file names already present in the dataset.

  Args:
    dataset: COCO dataset.

  Returns:
    Set of existing file_name values.
  """
  return {
    str(image_rec["file_name"])
    for image_rec in dataset.get("images", [])
  }

def load_augmented_items_from_output_dataset(
  dataset: Dict[str, Any],
  images_dir: Path
) -> List[Tuple[np.ndarray, CocoImageRecord]]:
  """Load existing non-mosaic images from an output COCO dataset.

  Args:
    dataset: Existing output COCO dataset.
    images_dir: Root directory where image files are stored.

  Returns:
    List of image arrays and records for mosaic sampling.
  """
  ann_by_image: Dict[int, List[CocoBox]] = {}
  for ann in dataset.get("annotations", []):
    image_id = int(ann["image_id"])
    bbox = [float(v) for v in ann["bbox"]]
    ann_by_image.setdefault(image_id, []).append(
      CocoBox(
        bbox = bbox,
        category_id = int(ann["category_id"]),
        annotation_id = int(ann.get("id", -1)),
        iscrowd = int(ann.get("iscrowd", 0)),
        area = float(ann.get("area", bbox[2] * bbox[3])),
        segmentation = copy.deepcopy(ann.get("segmentation")),
        source_image_id = image_id,
      )
    )
  items: List[Tuple[np.ndarray, CocoImageRecord]] = []
  for img in dataset.get("images", []):
    file_name = str(img["file_name"])
#
# skip existing mosaics; only use regular images as mosaic sources
#
    if file_name.startswith("mosaic/") or "/mosaic/" in file_name:
      continue
    image_path = images_dir / file_name
    if not image_path.exists():
      continue
    image = read_image(image_path)
    record = CocoImageRecord(
      image_id = int(img["id"]),
      file_name = file_name,
      width = int(img["width"]),
      height = int(img["height"]),
      annotations = copy.deepcopy(
        ann_by_image.get(int(img["id"]), [])
      ),
    )
#
# only use images that actually have at least one box
#
    if len(record.annotations) == 0:
      continue
    items.append((image, record))
  return items

def read_image(path: Path) -> np.ndarray:
  """Read image with OpenCV and convert to RGB.

  Args:
    path: Image path.

  Returns:
    RGB uint8 image.
  """
  image = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if image is None:
    raise FileNotFoundError(f"could not read image: {path}")
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def write_image(
  image: np.ndarray,
  path: Path,
  image_format: str,
  jpeg_quality: int
) -> None:
  """Write image to disk.

  Args:
    image: RGB uint8 image.
    path: Output file path.
    image_format: jpg or png.
    jpeg_quality: JPEG quality when jpg.
  """
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

def coco_to_records(coco: Dict[str, Any]) -> Tuple[List[CocoImageRecord], Dict[str, Any]]:
  """Convert raw COCO dict into image records.

  Args:
    coco: Raw COCO dataset.

  Returns:
    Tuple of image records and passthrough metadata.
  """
  images = coco.get("images", [])
  annotations = coco.get("annotations", [])
  ann_by_image: Dict[int, List[CocoBox]] = {}
  for ann in annotations:
    image_id = int(ann["image_id"])
    box = CocoBox(
      bbox = [float(v) for v in ann["bbox"]],
      category_id = int(ann["category_id"]),
      annotation_id = int(ann.get("id", -1)),
      iscrowd = int(ann.get("iscrowd", 0)),
      area = float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])),
      segmentation = copy.deepcopy(ann.get("segmentation")),
      source_image_id = image_id,
    )
    ann_by_image.setdefault(image_id, []).append(box)
  records: List[CocoImageRecord] = []
  for img in images:
    image_id = int(img["id"])
    record = CocoImageRecord(
      image_id = image_id,
      file_name = img["file_name"],
      width = int(img["width"]),
      height = int(img["height"]),
      annotations = ann_by_image.get(image_id, []),
    )
    records.append(record)
  meta = {
    "info": copy.deepcopy(coco.get("info", {})),
    "licenses": copy.deepcopy(coco.get("licenses", [])),
    "categories": copy.deepcopy(coco.get("categories", [])),
  }
  return records, meta

def build_category_lookup(
  categories: Sequence[Dict[str, Any]]
) -> Tuple[Dict[int, str], Dict[str, int]]:
  """Build category lookup maps.

  Args:
    categories: COCO category records.

  Returns:
    Tuple of id->name and lowercased name->id maps.
  """
  id_to_name: Dict[int, str] = {}
  name_to_id: Dict[str, int] = {}
  for cat in categories:
    cat_id = int(cat["id"])
    cat_name = str(cat["name"])
    id_to_name[cat_id] = cat_name
    name_to_id[cat_name.lower()] = cat_id
  return id_to_name, name_to_id

def resolve_class_filter(
  class_tokens: Optional[Sequence[str]],
  categories: Sequence[Dict[str, Any]]
) -> Optional[set]:
  """Resolve requested class names or ids to category ids.

  Args:
    class_tokens: User-supplied class tokens.
    categories: COCO category records.

  Returns:
    Set of category ids or None.
  """
  if not class_tokens:
    return None
  id_to_name, name_to_id = build_category_lookup(categories)
  resolved: set = set()
  unknown: List[str] = []
  for token in class_tokens:
    token_s = str(token).strip()
    if token_s == "":
      continue
    if token_s.isdigit():
      cat_id = int(token_s)
      if cat_id in id_to_name:
        resolved.add(cat_id)
      else:
        unknown.append(token_s)
      continue
    cat_id = name_to_id.get(token_s.lower())
    if cat_id is None:
      unknown.append(token_s)
      continue
    resolved.add(cat_id)
  if unknown:
    raise ValueError(
      "unknown classes in --classes-to-augment: " + ", ".join(unknown)
    )
  if not resolved:
    raise ValueError("--classes-to-augment resolved to no valid classes")
  return resolved

def record_matches_class_filter(
  record: CocoImageRecord,
  class_ids: Optional[set]
) -> bool:
  """Check whether a record should be augmented.

  Args:
    record: Image record.
    class_ids: Optional allowed category ids.

  Returns:
    True when the record matches.
  """
  if class_ids is None:
    return True
  ann_class_ids = {int(ann.category_id) for ann in record.annotations}
  return len(ann_class_ids.intersection(class_ids)) > 0

def build_single_image_transform(
  min_area: float,
  min_visibility: float
) -> A.Compose:
  """Build bbox-aware full-frame augmentation pipeline.

  Args:
    min_area: Minimum bbox area to keep.
    min_visibility: Minimum visibility fraction.

  Returns:
    Albumentations compose object.
  """
  return A.Compose(
    [
      A.HorizontalFlip(p = 0.50),
      A.Affine(
        scale = (0.95, 1.20),
        translate_percent = {
          "x": (-0.04, 0.04),
          "y": (-0.04, 0.04),
        },
        rotate = (-8, 8),
        shear = (-4, 4),
        fit_output = False,
        p = 0.60,
      ),
      A.OneOf(
        [
          A.RandomBrightnessContrast(
            brightness_limit = 0.20,
            contrast_limit = 0.20,
            p = 1.0,
          ),
          A.CLAHE(
            clip_limit = (1.0, 2.5),
            p = 1.0,
          ),
          A.HueSaturationValue(
            hue_shift_limit = 4,
            sat_shift_limit = 10,
            val_shift_limit = 10,
            p = 1.0,
          ),
        ],
        p = 0.70,
      ),
      A.OneOf(
        [
          A.GaussianBlur(
            blur_limit = (3, 3),
            p = 1.0,
          ),
          A.GaussNoise(
            std_range = (0.01, 0.035),
            p = 1.0,
          ),
          A.MotionBlur(
            blur_limit = (3, 3),
            p = 1.0,
          ),
          A.NoOp(p = 1.0),
        ],
        p = 0.20,
      ),
    ],
    bbox_params = A.BboxParams(
      format = "coco",
      label_fields = [
        "category_ids",
        "iscrowd_flags",
        "ann_ids",
        "src_ids",
      ],
      min_area = min_area,
      min_visibility = min_visibility,
      clip = True,
    ),
  )

def stem_from_filename(file_name: str) -> str:
  """Return safe stem from file name.

  Args:
    file_name: Relative file name.

  Returns:
    Stem string.
  """
  return Path(file_name).stem

def make_output_file_name(
  prefix: str,
  stem: str,
  idx: int,
  image_format: str
) -> str:
  """Build output file name.

  Args:
    prefix: Output prefix.
    stem: Source stem.
    idx: Running index.
    image_format: jpg or png.

  Returns:
    File name.
  """
  return f"{prefix}_{stem}_{idx:06d}.{image_format}"

def get_source_folder(file_name: str) -> str:
  """Return source folder relative path or empty string.

  Args:
    file_name: COCO file_name value.

  Returns:
    Parent folder path or empty string when none exists.
  """
  parent = Path(file_name).parent
  if str(parent) == ".":
    return ""
  return parent.as_posix()

def build_output_relative_path(
  out_name: str,
  record: CocoImageRecord,
  organize_by_source_folder: bool
) -> str:
  """Build relative output path for an image.

  Args:
    out_name: Output file name only.
    record: Source image record.
    organize_by_source_folder: Whether to preserve source folder grouping.

  Returns:
    Relative path from output root.
  """
  if not organize_by_source_folder:
    return f"{out_name}"
  folder = get_source_folder(record.file_name)
  if folder:
    return f"{folder}/{out_name}"
  return f"{out_name}"

def copy_original_image(
  src_path: Path,
  dst_path: Path
) -> None:
  """Copy an original image into the output dataset.

  Args:
    src_path: Source image path.
    dst_path: Destination image path.
  """
  ensure_dir(dst_path.parent)
  shutil.copy2(src_path, dst_path)

def transform_one_image(
  image: np.ndarray,
  record: CocoImageRecord,
  transform: A.Compose,
  rng: random.Random,
  allow_negative_samples: bool,
  max_trials_per_output: int
) -> Tuple[np.ndarray, List[CocoBox]]:
  """Apply single-image transform with bbox retry logic.

  Args:
    image: Source RGB image.
    record: Source image record.
    transform: Albumentations transform.
    rng: Random generator.
    allow_negative_samples: Whether zero-box outputs are accepted.
    max_trials_per_output: Retry count when boxes disappear.

  Returns:
    Augmented image and annotation list.
  """
  bboxes = [ann.bbox for ann in record.annotations]
  category_ids = [ann.category_id for ann in record.annotations]
  iscrowd_flags = [ann.iscrowd for ann in record.annotations]
  ann_ids = [ann.annotation_id for ann in record.annotations]
  src_ids = [ann.source_image_id for ann in record.annotations]
  segmentations = [copy.deepcopy(ann.segmentation) for ann in record.annotations]
  source_areas = [ann.area for ann in record.annotations]
  last_result: Optional[Dict[str, Any]] = None
  for _ in range(max_trials_per_output):
    seed_value = rng.randint(0, 2 ** 31 - 1)
    random.seed(seed_value)
    np.random.seed(seed_value)
    result = transform(
      image = image,
      bboxes = bboxes,
      category_ids = category_ids,
      iscrowd_flags = iscrowd_flags,
      ann_ids = ann_ids,
      src_ids = src_ids,
    )
    last_result = result
    if allow_negative_samples or len(result["bboxes"]) > 0:
      break
  if last_result is None:
    raise RuntimeError("transform failed to produce a result")
  out_boxes: List[CocoBox] = []
  kept_count = len(last_result["bboxes"])
  for i in range(kept_count):
    bbox = [float(v) for v in last_result["bboxes"][i]]
    out_boxes.append(
      CocoBox(
        bbox = bbox,
        category_id = int(last_result["category_ids"][i]),
        annotation_id = int(last_result["ann_ids"][i]),
        iscrowd = int(last_result["iscrowd_flags"][i]),
        area = float(bbox[2] * bbox[3]),
        segmentation = None,
        source_image_id = int(last_result["src_ids"][i]),
      )
    )
  return last_result["image"], out_boxes

def resize_with_aspect(
  image: np.ndarray,
  max_w: int,
  max_h: int
) -> Tuple[np.ndarray, float, float]:
  """Resize image to fit within slot while preserving aspect.

  Args:
    image: RGB image.
    max_w: Max output width.
    max_h: Max output height.

  Returns:
    Resized image and x/y scale factors.
  """
  h, w = image.shape[:2]
  scale = min(max_w / max(w, 1), max_h / max(h, 1))
  new_w = max(1, int(round(w * scale)))
  new_h = max(1, int(round(h * scale)))
  resized = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
  sx = new_w / max(w, 1)
  sy = new_h / max(h, 1)
  return resized, sx, sy

def choose_grid(n_images: int) -> Tuple[int, int]:
  """Choose mosaic grid layout.

  Args:
    n_images: Number of source images.

  Returns:
    Grid as rows, cols.
  """
#
# for two-image mosaics, preserve more horizontal context by placing
# images side-by-side instead of stacking them.
#
  if n_images == 2:
    return 1, 2
  cols = math.ceil(math.sqrt(n_images))
  rows = math.ceil(n_images / cols)
  return rows, cols

def clip_bbox_to_canvas(
  bbox: Sequence[float],
  canvas_w: int,
  canvas_h: int,
  min_area: float
) -> Optional[List[float]]:
  """Clip COCO bbox to image canvas.

  Args:
    bbox: Box [x, y, w, h].
    canvas_w: Canvas width.
    canvas_h: Canvas height.
    min_area: Minimum area to keep.

  Returns:
    Clipped bbox or None.
  """
  x, y, w, h = bbox
  x1 = max(0.0, x)
  y1 = max(0.0, y)
  x2 = min(float(canvas_w), x + w)
  y2 = min(float(canvas_h), y + h)
  new_w = x2 - x1
  new_h = y2 - y1
  if new_w <= 0 or new_h <= 0:
    return None
  if new_w * new_h < min_area:
    return None
  return [x1, y1, new_w, new_h]

def build_mosaic(
  items: Sequence[Tuple[np.ndarray, CocoImageRecord]],
  mosaic_width: int,
  mosaic_height: int,
  min_area: float,
  rng: random.Random,
) -> Tuple[np.ndarray, List[CocoBox]]:
  """Build a simple grid mosaic from 2..N images.

  Args:
    items: Sequence of image arrays and records.
    mosaic_width: Output width.
    mosaic_height: Output height.
    min_area: Minimum bbox area to keep.
    rng: Random generator.

  Returns:
    Mosaic image and updated boxes.
  """
  canvas = np.zeros((mosaic_height, mosaic_width, 3), dtype = np.uint8)
  rows, cols = choose_grid(len(items))
  cell_w = mosaic_width // cols
  cell_h = mosaic_height // rows
  out_boxes: List[CocoBox] = []
  for idx, (image, record) in enumerate(items):
    row = idx // cols
    col = idx % cols
    x0 = col * cell_w
    y0 = row * cell_h
    x1 = mosaic_width if col == cols - 1 else (col + 1) * cell_w
    y1 = mosaic_height if row == rows - 1 else (row + 1) * cell_h
    slot_w = x1 - x0
    slot_h = y1 - y0
    resized, sx, sy = resize_with_aspect(image, slot_w, slot_h)
    rh, rw = resized.shape[:2]
    pad_x = max(0, (slot_w - rw) // 2)
    pad_y = max(0, (slot_h - rh) // 2)
    dst_x = x0 + pad_x
    dst_y = y0 + pad_y
    canvas[dst_y:dst_y + rh, dst_x:dst_x + rw] = resized
    for ann in record.annotations:
      bx, by, bw, bh = ann.bbox
      new_bbox = [
        float(dst_x + bx * sx),
        float(dst_y + by * sy),
        float(bw * sx),
        float(bh * sy),
      ]
      clipped = clip_bbox_to_canvas(
        bbox = new_bbox,
        canvas_w = mosaic_width,
        canvas_h = mosaic_height,
        min_area = min_area,
      )
      if clipped is None:
        continue
      out_boxes.append(
        CocoBox(
          bbox = clipped,
          category_id = ann.category_id,
          annotation_id = ann.annotation_id,
          iscrowd = ann.iscrowd,
          area = float(clipped[2] * clipped[3]),
          segmentation = None,
          source_image_id = ann.source_image_id,
        )
      )
  return canvas, out_boxes

def build_output_dataset(meta: Dict[str, Any]) -> Dict[str, Any]:
  """Initialize output COCO dataset.

  Args:
    meta: Metadata dict.

  Returns:
    Output COCO dict.
  """
  info = copy.deepcopy(meta.get("info", {}))
  info["description"] = (
    str(info.get("description", "")).strip() +
    " | augmented with Albumentations and optional mosaic"
  ).strip(" |")
  return {
    "info": info,
    "licenses": copy.deepcopy(meta.get("licenses", [])),
    "categories": copy.deepcopy(meta.get("categories", [])),
    "images": [],
    "annotations": [],
  }

def validate_source_paths(records: Sequence[CocoImageRecord], images_dir: Path) -> None:
  """Validate that all source image files exist.

  Args:
    records: Source image records.
    images_dir: Images root.
  """
  missing: List[str] = []
  for record in records:
    path = images_dir / record.file_name
    if not path.exists():
      missing.append(record.file_name)
  if missing:
    sample = ", ".join(missing[:10])
    raise FileNotFoundError(
      f"missing {len(missing)} source images under {images_dir}. sample: {sample}"
    )
#
def filter_records_to_existing_paths(
  records: Sequence[CocoImageRecord],
  images_dir: Path
) -> List[CocoImageRecord]:
  """Keep only records whose files exist under images_dir.

  Args:
    records: Candidate image records.
    images_dir: Root source image directory.

  Returns:
    Records with existing source files.
  """
  existing_records: List[CocoImageRecord] = []
  missing: List[str] = []
  for record in records:
    path = images_dir / record.file_name
    if path.exists():
      existing_records.append(record)
    else:
      missing.append(record.file_name)
#
# report missing files but continue
#
  if missing:
    sample = ", ".join(missing[:10])
    print(
      f"warning: skipping {len(missing)} records missing under "
      f"{images_dir}. sample: {sample}"
    )
  return existing_records
#
def append_output_image(
  dataset: Dict[str, Any],
  file_name: str,
  width: int,
  height: int,
  image_id: int
) -> None:
  """Append image entry to COCO dataset.

  Args:
    dataset: Output COCO dict.
    file_name: Relative output file name.
    width: Image width.
    height: Image height.
    image_id: New image id.
  """
  dataset["images"].append(
    {
      "id": image_id,
      "file_name": file_name,
      "width": width,
      "height": height,
    }
  )

def append_output_annotations(
  dataset: Dict[str, Any],
  boxes: Sequence[CocoBox],
  image_id: int,
  next_annotation_id: int
) -> int:
  """Append annotation entries and return next annotation id.

  Args:
    dataset: Output COCO dict.
    boxes: Boxes to append.
    image_id: Output image id.
    next_annotation_id: Starting annotation id.

  Returns:
    Next unused annotation id.
  """
  ann_id = next_annotation_id
  for box in boxes:
    dataset["annotations"].append(
      {
        "id": ann_id,
        "image_id": image_id,
        "category_id": int(box.category_id),
        "bbox": [round(float(v), 3) for v in box.bbox],
        "area": round(float(box.bbox[2] * box.bbox[3]), 3),
        "iscrowd": int(box.iscrowd),
      }
    )
    ann_id += 1
  return ann_id

def main() -> None:
  """Run augmentation pipeline."""
  args = parse_args()
  rng = random.Random(args.seed)
  np.random.seed(args.seed)
  coco_json = Path(args.coco_json).resolve()
  images_dir = Path(args.images_dir).resolve()
  output_dir = Path(args.output_dir).resolve()
  output_images_dir = output_dir
  output_json_path = (
    Path(args.output_json).resolve()
    if args.output_json is not None
    else output_dir / "annotations.json"
  )
  ensure_dir(output_dir)
  ensure_dir(output_images_dir)
  ensure_dir(output_json_path.parent)
  coco = load_json(coco_json)
  records, meta = coco_to_records(coco)
#
# load or initialize output dataset
#
  existing_out_dataset = None
  if args.append_output:
    existing_out_dataset = load_existing_output_dataset(output_json_path)
    if existing_out_dataset is None:
      raise FileNotFoundError(
        f"append target json not found: {output_json_path}"
      )
  if existing_out_dataset is not None:
    out_dataset = existing_out_dataset
    next_image_id, next_annotation_id = get_next_ids_from_dataset(
      out_dataset
    )
    running_idx = get_next_running_index_from_dataset(out_dataset)
  else:
    out_dataset = build_output_dataset(meta)
    next_image_id = 1
    next_annotation_id = 1
    running_idx = 1
  existing_file_names = build_existing_file_name_set(out_dataset)
#
# validate high-level mode combinations
#
  if args.mosaics_only and not args.append_output:
    raise ValueError("--mosaics-only requires --append-output")
#
# resolve class filter
#
  class_filter_ids = resolve_class_filter(
    class_tokens = args.classes_to_augment,
    categories = meta.get("categories", []),
  )
#
# choose augmentation source pool
#
  if args.mosaics_only:
    records_to_augment = []
    augmented_items = load_augmented_items_from_output_dataset(
      dataset = out_dataset,
      images_dir = images_dir,
    )
    if class_filter_ids is not None:
      filtered_items = []
      for image_arr, record in augmented_items:
        if record_matches_class_filter(record, class_filter_ids):
          filtered_items.append((image_arr, record))
      augmented_items = filtered_items
    if len(augmented_items) == 0:
      raise FileNotFoundError(
        "no eligible existing output images found for --mosaics-only"
      )
  else:
    records_to_augment = [
      record for record in records
      if record_matches_class_filter(record, class_filter_ids)
    ]
    records_to_augment = filter_records_to_existing_paths(
      records = records_to_augment,
      images_dir = images_dir,
    )
    if not records_to_augment:
      raise FileNotFoundError(
        "no selected source images exist under "
        f"{images_dir}"
      )
    transform = build_single_image_transform(
      min_area = args.min_area,
      min_visibility = args.min_visibility,
    )
    augmented_items = []
#
# optionally copy originals
#
  if args.include_originals and not args.mosaics_only:
    for record in records_to_augment:
      source_path = images_dir / record.file_name
      src_rel_path = build_output_relative_path(
        out_name = Path(record.file_name).name,
        record = record,
        organize_by_source_folder = args.organize_by_source_folder,
      )
      if src_rel_path in existing_file_names:
        continue
      dst_path = output_dir / src_rel_path
      copy_original_image(source_path, dst_path)
      append_output_image(
        dataset = out_dataset,
        file_name = src_rel_path,
        width = record.width,
        height = record.height,
        image_id = next_image_id,
      )
      existing_file_names.add(src_rel_path)
      next_annotation_id = append_output_annotations(
        dataset = out_dataset,
        boxes = record.annotations,
        image_id = next_image_id,
        next_annotation_id = next_annotation_id,
      )
      next_image_id += 1
#
# generate single-image augmentations
#
  if not args.mosaics_only:
    for record in records_to_augment:
      source_path = images_dir / record.file_name
      image = read_image(source_path)
      for copy_idx in range(args.copies_per_image):
        aug_image, aug_boxes = transform_one_image(
          image = image,
          record = record,
          transform = transform,
          rng = rng,
          allow_negative_samples = args.allow_negative_samples,
          max_trials_per_output = args.max_trials_per_output,
        )
        out_name = make_output_file_name(
          prefix = "aug",
          stem = stem_from_filename(record.file_name),
          idx = running_idx,
          image_format = args.image_format,
        )
        running_idx += 1
        out_rel_path = build_output_relative_path(
          out_name = out_name,
          record = record,
          organize_by_source_folder = args.organize_by_source_folder,
        )
        out_abs_path = output_dir / out_rel_path
        ensure_dir(out_abs_path.parent)
        write_image(
          image = aug_image,
          path = out_abs_path,
          image_format = args.image_format,
          jpeg_quality = args.jpeg_quality,
        )
        h, w = aug_image.shape[:2]
        append_output_image(
          dataset = out_dataset,
          file_name = out_rel_path,
          width = w,
          height = h,
          image_id = next_image_id,
        )
        existing_file_names.add(out_rel_path)
        next_annotation_id = append_output_annotations(
          dataset = out_dataset,
          boxes = aug_boxes,
          image_id = next_image_id,
          next_annotation_id = next_annotation_id,
        )
        augmented_items.append(
          (
            aug_image,
            CocoImageRecord(
              image_id = next_image_id,
              file_name = out_rel_path,
              width = w,
              height = h,
              annotations = copy.deepcopy(aug_boxes),
            ),
          )
        )
        next_image_id += 1
#
# generate mosaics
#
  if args.enable_mosaic:
    if args.min_mosaic_images < 2:
      raise ValueError("--min-mosaic-images must be >= 2")
    if args.max_mosaic_images < args.min_mosaic_images:
      raise ValueError(
        "--max-mosaic-images must be >= --min-mosaic-images"
      )
    if len(augmented_items) < args.min_mosaic_images:
      raise ValueError(
        "not enough augmented items available to create mosaics"
      )
    source_count_for_mosaics = len(augmented_items)
    total_mosaics = max(
      1,
      int(round(source_count_for_mosaics * args.mosaic_copies))
    )
    mosaic_w, mosaic_h = args.mosaic_size
    for _ in range(total_mosaics):
      max_pick = min(args.max_mosaic_images, len(augmented_items))
      if max_pick < args.min_mosaic_images:
        break
      n_pick = rng.randint(args.min_mosaic_images, max_pick)
      chosen = rng.sample(augmented_items, k = n_pick)
      mosaic_image, mosaic_boxes = build_mosaic(
        items = chosen,
        mosaic_width = mosaic_w,
        mosaic_height = mosaic_h,
        min_area = args.min_area,
        rng = rng,
      )
      out_name = make_output_file_name(
        prefix = "mosaic",
        stem = "mix",
        idx = running_idx,
        image_format = args.image_format,
      )
      running_idx += 1
      out_rel_path = f"mosaic/{out_name}"
      out_abs_path = output_dir / out_rel_path
      ensure_dir(out_abs_path.parent)
      write_image(
        image = mosaic_image,
        path = out_abs_path,
        image_format = args.image_format,
        jpeg_quality = args.jpeg_quality,
      )
      append_output_image(
        dataset = out_dataset,
        file_name = out_rel_path,
        width = mosaic_w,
        height = mosaic_h,
        image_id = next_image_id,
      )
      existing_file_names.add(out_rel_path)
      next_annotation_id = append_output_annotations(
        dataset = out_dataset,
        boxes = mosaic_boxes,
        image_id = next_image_id,
        next_annotation_id = next_annotation_id,
      )
      next_image_id += 1
#
# save outputs and report summary
#
  save_json(out_dataset, output_json_path)
  print(f"wrote images to: {output_images_dir}")
  print(f"wrote annotations to: {output_json_path}")
  print(f"source images total: {len(records)}")
  print(f"source images selected for augmentation: {len(records_to_augment)}")
  print(f"mosaic source items available: {len(augmented_items)}")
  print(f"images: {len(out_dataset['images'])}")
  print(f"annotations: {len(out_dataset['annotations'])}")

if __name__ == "__main__":
  main()
