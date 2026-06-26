#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:35:34 2026

@author: eafpres
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

#
# constants
#

WELD_CATEGORY = {
  "id": 1,
  "name": "weld_region",
  "supercategory": None,
}

#
# helpers
#

def parse_args() -> argparse.Namespace:
  """Parse command-line arguments.

  Returns:
    argparse.Namespace: Parsed args.
  """
  parser = argparse.ArgumentParser(
    description = "Convert multi-class COCO JSON to one-class weld-region COCO"
  )
  parser.add_argument(
    "--input-json",
    required = True,
    type = str,
    help = "Path to source COCO JSON"
  )
  parser.add_argument(
    "--output-json",
    required = True,
    type = str,
    help = "Path to output COCO JSON"
  )
  parser.add_argument(
    "--good-json",
    required = False,
    default = None,
    type = str,
    help = (
      "Optional COCO JSON containing good weld annotations to merge before "
      "flattening."
    )
  )
  parser.add_argument(
    "--drop-images-without-annotations",
    action = "store_true",
    help = (
      "Drop images that have no annotations after conversion. "
      "By default they are kept as negatives."
    )
  )
  return parser.parse_args()

def load_json(path: Path) -> Dict[str, Any]:
  """Load JSON from disk.

  Args:
    path: JSON path.

  Returns:
    Dict[str, Any]: Parsed JSON.
  """
  with path.open("r", encoding = "utf-8") as f:
    return json.load(f)

def save_json(data: Dict[str, Any], path: Path) -> None:
  """Write JSON to disk.

  Args:
    data: Output payload.
    path: Output path.
  """
  path.parent.mkdir(parents = True, exist_ok = True)
  with path.open("w", encoding = "utf-8") as f:
    json.dump(data, f, indent = 2)

def make_ann_key(ann: Dict[str, Any]) -> str:
  """Create a stable key for duplicate annotation detection.

  Args:
    ann: COCO annotation.

  Returns:
    str: Stable JSON key excluding annotation id.
  """
  payload = {
    "image_id": int(ann["image_id"]),
    "category_id": int(ann.get("category_id", -1)),
    "bbox": [round(float(v), 6) for v in ann.get("bbox", [])],
    "area": round(float(ann.get("area", 0.0)), 6),
    "iscrowd": int(ann.get("iscrowd", 0)),
    "segmentation": ann.get("segmentation", None),
  }
  return json.dumps(payload, sort_keys = True)

def get_next_id(items: List[Dict[str, Any]]) -> int:
  """Get the next positive integer id for a COCO item list.

  Args:
    items: COCO images or annotations.

  Returns:
    int: Next available id.
  """
  if not items:
    return 1
  return max(int(item["id"]) for item in items) + 1

def merge_good_annotations(
  coco: Dict[str, Any],
  good_coco: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
  """Merge good weld annotations into a source COCO payload.

  Args:
    coco: Main COCO payload.
    good_coco: COCO payload with good weld annotations.

  Returns:
    Tuple[Dict[str, Any], Dict[str, int]]: Merged COCO and report counts.
  """
  out = copy.deepcopy(coco)
  out.setdefault("images", [])
  out.setdefault("annotations", [])
  images = out["images"]
  annotations = out["annotations"]
  existing_image_ids = {int(img["id"]) for img in images}
  existing_ann_keys = {make_ann_key(ann) for ann in annotations}
  image_by_file_name = {
    img.get("file_name"): img
    for img in images
    if img.get("file_name") is not None
  }
  next_image_id = get_next_id(images)
  next_ann_id = get_next_id(annotations)
  image_id_map: Dict[int, int] = {}
  added_images = 0
  matched_images = 0
  remapped_images = 0
  added_annotations = 0
  skipped_duplicate_annotations = 0
  skipped_missing_images = 0

#
# map or add good images
#

  for good_img in good_coco.get("images", []):
    good_image_id = int(good_img["id"])
    file_name = good_img.get("file_name")
    if file_name in image_by_file_name:
      image_id_map[good_image_id] = int(image_by_file_name[file_name]["id"])
      matched_images += 1
      continue
    img_copy = copy.deepcopy(good_img)
    if good_image_id in existing_image_ids:
      img_copy["id"] = next_image_id
      image_id_map[good_image_id] = next_image_id
      existing_image_ids.add(next_image_id)
      next_image_id += 1
      remapped_images += 1
    else:
      image_id_map[good_image_id] = good_image_id
      existing_image_ids.add(good_image_id)
    images.append(img_copy)
    if img_copy.get("file_name") is not None:
      image_by_file_name[img_copy["file_name"]] = img_copy
    added_images += 1

#
# add good annotations with remapped image and annotation ids
#

  for good_ann in good_coco.get("annotations", []):
    old_image_id = int(good_ann["image_id"])
    if old_image_id not in image_id_map:
      skipped_missing_images += 1
      continue
    ann_copy = copy.deepcopy(good_ann)
    ann_copy["image_id"] = image_id_map[old_image_id]
    ann_key = make_ann_key(ann_copy)
    if ann_key in existing_ann_keys:
      skipped_duplicate_annotations += 1
      continue
    ann_copy["id"] = next_ann_id
    annotations.append(ann_copy)
    existing_ann_keys.add(ann_key)
    next_ann_id += 1
    added_annotations += 1
  report = {
    "good_images_matched_by_file_name": matched_images,
    "good_images_added": added_images,
    "good_images_remapped_due_to_id_collision": remapped_images,
    "good_annotations_added": added_annotations,
    "good_annotations_skipped_as_duplicates": skipped_duplicate_annotations,
    "good_annotations_skipped_missing_image": skipped_missing_images,
  }
  return out, report

def convert_to_weld_region(
  coco: Dict[str, Any],
  drop_images_without_annotations: bool,
) -> Dict[str, Any]:
  """Convert a multi-class COCO file to one-class weld-region COCO.

  Args:
    coco: Source COCO dict.
    drop_images_without_annotations: Whether to remove images with no anns.

  Returns:
    Dict[str, Any]: Converted COCO dict.
  """
  out: Dict[str, Any] = {
    "info": copy.deepcopy(coco.get("info", {})),
    "licenses": copy.deepcopy(coco.get("licenses", [])),
    "categories": [copy.deepcopy(WELD_CATEGORY)],
    "images": [],
    "annotations": [],
  }

#
# convert annotations to one class
#

  kept_image_ids: Set[int] = set()
  next_ann_id = 1
  for ann in coco.get("annotations", []):
    image_id = int(ann["image_id"])
    bbox = [float(v) for v in ann["bbox"]]
    area = float(ann.get("area", bbox[2] * bbox[3]))
    out_ann = {
      "id": next_ann_id,
      "image_id": image_id,
      "category_id": 1,
      "bbox": bbox,
      "area": area,
      "iscrowd": int(ann.get("iscrowd", 0)),
    }
    if "segmentation" in ann:
      out_ann["segmentation"] = copy.deepcopy(ann["segmentation"])
    out["annotations"].append(out_ann)
    kept_image_ids.add(image_id)
    next_ann_id += 1

#
# keep or filter images
#

  for img in coco.get("images", []):
    image_id = int(img["id"])
    if drop_images_without_annotations and image_id not in kept_image_ids:
      continue
    out["images"].append(copy.deepcopy(img))
  return out

def main() -> None:
  """Run conversion."""
  args = parse_args()
  input_json = Path(args.input_json).resolve()
  output_json = Path(args.output_json).resolve()
  good_json: Optional[Path] = None
  coco = load_json(input_json)
  merge_report: Dict[str, int] = {}

#
# optionally merge good weld annotations before flattening
#

  if args.good_json is not None:
    good_json = Path(args.good_json).resolve()
    good_coco = load_json(good_json)
    coco, merge_report = merge_good_annotations(
      coco = coco,
      good_coco = good_coco,
    )
  out = convert_to_weld_region(
    coco = coco,
    drop_images_without_annotations = args.drop_images_without_annotations,
  )
  save_json(out, output_json)

#
# report
#

  print(f"input json: {input_json}")
  if good_json is not None:
    print(f"good json: {good_json}")
    for key, value in merge_report.items():
      print(f"{key}: {value}")
  print(f"output json: {output_json}")
  print(f"images: {len(out['images'])}")
  print(f"annotations: {len(out['annotations'])}")
  print(f"categories: {out['categories']}")

if __name__ == "__main__":
  main()